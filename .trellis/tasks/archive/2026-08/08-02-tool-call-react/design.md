# Technical Design: Chem-LGBO Tool Call + ReAct

## Decision

在现有 `_llm_mean_shift()` 内做一个两次上限的局部循环，复用现有 Parser 和候选池 mask 校验。没有新抽象层、没有第二套 schema validator。

## Data Flow

```text
ChemLGBOEngine._llm_mean_shift
  -> LGBOEngine._call_llm(messages, tools, tool_choice)
    -> DeepSeekClient.chat(..., temperature=self.llm_temperature)
      -> LlmCallResult(content, tool_calls)
  -> 提取唯一目标 tool arguments
  -> parse_subspace_response
  -> build_subspace_mask & remaining
  -> accepted: masked_mean_shift
     retryable failure: 追加 assistant/tool 回执后再调用一次
     transport failure or second failure: existing GP fallback
```

## 1. Client Contract

`LlmCallResult` 追加默认字段：

```python
tool_calls: list[dict[str, Any]] | None = None
```

`DeepSeekClient.chat`：

```python
def chat(
    self,
    messages: list[dict[str, Any]],
    max_tokens: int = 2048,
    extra_body: dict[str, Any] | None = None,
    *,
    temperature: float = 0.0,
) -> LlmCallResult:
```

保留 `extra_body` 的位置兼容性，将新参数设为 keyword-only，避免现有第三位置调用被误解释为 temperature。

响应规则：

```python
message = choice.get("message", {})
content = message.get("content") or ""
tool_calls = message.get("tool_calls") or None
if not content.strip() and not tool_calls:
    # existing empty-response failure
```

payload 的 `temperature` 来自显式参数；`extra_body` 不得覆盖该受保护 key。

## 2. Base Engine

`LGBOEngine.__init__(..., llm_temperature: float = 0.0)` 保存为 float。`_call_llm()` 增加 keyword-only `tools`/`tool_choice`，只在非空时写入 `extra_body`，并调用：

```python
self._client.chat(
    messages,
    extra_body=extra_body or None,
    temperature=self.llm_temperature,
)
```

现有非 Chem 调用保持空 tools 和 0.0 温度。

## 3. Tool Schema

schema 定义为模块级常量，package 与 submission 各自保持同一字面值：

```python
PROPOSE_SUBSPACE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_sparse_subspace",
        "description": "Propose a sparse categorical subspace for the remaining experiment pool.",
        "parameters": {
            "type": "object",
            "properties": {
                "subspace": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "minProperties": 1,
                }
            },
            "required": ["subspace"],
            "additionalProperties": False,
        },
    },
}
```

`tool_choice` 强制指定该函数。option enum 仍不进入 schema。

## 4. Attempt Evaluation

在 `_llm_mean_shift()` 中预先计算 `remaining`、`candidate_features`。每次结果依次检查：

1. transport/client status；
2. `tool_calls` 存在；
3. 恰有一个目标函数调用；
4. arguments 是字符串；
5. 现有 Parser 接受；
6. `raw_size > 0`；
7. `0 < mask_size < remaining_pool_size`。

返回一个局部 attempt 结果即可；不新增公共类型。第一轮失败且属于 R4 列表时，构造重试消息。第二轮复用完全相同的检查。

### Tool Error Payload

最小 JSON：

```json
{
  "status": "rejected",
  "reason": "already_queried_only",
  "constraint": "Choose a subspace containing at least one unqueried candidate and not the entire remaining pool."
}
```

Parser 错误按 reason 映射到简短纠正说明；不把完整候选池、oracle 值或 trajectory 重复写入回执。

### Missing/Malformed Tool Calls

- 恰好一个目标调用且存在 `id`：回执使用该 `id`。
- 完全没有调用、函数名错误、调用数不为一或缺少 `id`：无法构造无歧义且合法的 `role=tool` 回执；追加一条局部 `role=user` 纠正消息后重试。

## 5. Telemetry

扩展 `_store_guidance_artifact()` 与 `_fallback()` 的 keyword-only 参数：

```python
react_retried: bool = False
react_first_reason: str | None = None
llm_attempts: int = 1
tool_call_id: str | None = None
```

只写 `guidance_artifacts`。`trajectory` 继续只含已执行实验状态；下一轮 prompt 不见 retry 对话。

最终 `raw_response` 保存最后一次被解析的 arguments。第一轮错误内容通过 `react_first_reason` 统计，不复制整段响应。

## 6. Temperature and Provenance

`ChemLGBOEngine.__init__` 显式参数：

```python
llm_temperature: float = 0.2
```

传给 `super()`。实验 runner/config、prompt ablation `model_config` 和输出 provenance 从 engine/experiment 配置读取真实值，不再写死 `0.0`。同时记录 `response_mode` 与 `max_react_retries`。

## 7. Mirrored Runtime

先修改 package 实现并测试，再对 submission 三个镜像文件做相同最小改动。用定向测试验证两条 import surface，而不是依赖人工 diff。

## 8. Tests

最小行为测试：

1. client：空 content + tool_calls 成功；默认/显式 temperature payload；受保护 key 不可覆盖；
2. base LGBO：默认 0.0、tools 透传、旧 fake client 兼容更新；
3. Chem：首轮 `unknown_value` 自愈；首轮 `already_queried_only` 自愈；两轮失败 fallback；retry message 的 `tool_call_id` 与局部性；artifact telemetry；
4. experiment：provenance 记录 0.2/tool mode；
5. submission：对应 smoke/contract 测试。

真实 API smoke 只运行保存状态，不跑完整 226 次长实验。输出必须区分结构解析成功、候选池语义接受、重试成功和最终 fallback。

## Risks

- DeepSeek 可能返回空 content：client 必须以 tool_calls 判定成功。
- Fake clients 可能没有新 keyword 参数：测试替身必须跟随真实接口，生产代码不做 `TypeError` 降级重调，避免掩盖接口错误。
- Treatment 重复旧 subspace 的概率高：语义错误回执必须明确“剩余池”，否则 Tool Calling 本身不会降低真实 fallback。
- 提升 temperature 改变实验协议：provenance 必须记录，旧 artifact 不可与新结果混为同一 treatment。
