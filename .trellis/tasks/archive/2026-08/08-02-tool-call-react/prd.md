# Chem-LGBO Tool Call + ReAct 重构

## Goal

将 `ChemLGBOEngine._llm_mean_shift` 的输出通道从纯文本 JSON 升级为强制 Tool Calling，并增加最多 1 次、仅限当前 BO step 的纠错重试。目标是同时处理：

1. JSON/字段/字典校验失败；
2. 子空间合法但对当前剩余候选池不可用。

`temperature` 改为显式、可追溯参数；默认值仅对 Chem-LGBO 提升到 `0.2`，其他 `DeepSeekClient` 与 `LGBOEngine` 调用保持 `0.0`。

## Confirmed Baseline

- 最新真实 artifact `chem_lgbo_prompt_feedback.json` 含 226 条记录、25 次 fallback，实际 fallback 率为 **11.1%**。
- 25 次全部是 `already_queried_only`：Control 5 次、Treatment 20 次。当前主要问题不是 JSON 格式，而是建议子空间已经没有未查询候选。
- Tool Calling 探针验证了结构化参数传输，但尚未覆盖真实主循环中的剩余候选语义校验。
- LLM guidance 只修改 posterior mean：命中子空间的候选获得 `+1σ`；不硬过滤候选池。
- `packages/bo-core` 与 `Compitetion/submission/code/bo_core` 存在运行时镜像，必须同步修改。

## Requirements

### R1: Tool Schema

定义并强制调用 `propose_sparse_subspace`：

- 参数只含必填对象 `subspace`；
- `subspace` 允许任意已声明 feature key，每个值为非空字符串数组；
- 不把完整 option enum 塞入 schema，现有 Parser 继续负责 feature/value 精确校验；
- tool arguments 规范化为 `{"subspace": ...}` 后复用 `parse_subspace_response`，不建立第二套 Parser。

### R2: Client Contract

- `LlmCallResult` 新增 `tool_calls: list[dict[str, Any]] | None = None`，不破坏现有构造者。
- `DeepSeekClient.chat()` 新增关键字参数 `temperature: float = 0.0`。
- `messages` 类型允许 OpenAI-compatible 嵌套对象，不能继续限定为 `dict[str, str]`。
- 成功响应允许 `content == ""` 且存在 `tool_calls`；只有两者都为空才是空响应。
- `tools`/`tool_choice` 通过 `extra_body` 传递；`temperature` 只有显式参数一个来源。若 `extra_body` 含受保护的 `temperature`，沿用现有受保护字段拒绝策略，不能静默覆盖。

### R3: Base LGBO Compatibility

- `LGBOEngine.__init__` 新增 `llm_temperature: float = 0.0`。
- `_call_llm()` 只增加可选 `tools`、`tool_choice`，并始终传递 `self.llm_temperature`。
- 非 Chem 调用者、纯文本响应路径和默认温度行为不变。

### R4: One-Retry ReAct Loop

每个 Chem-LGBO step 最多调用两次 LLM：初次 + 1 次纠错。

可重试原因包括：

- 客户端返回成功但缺少目标 tool call；
- tool name 不匹配、tool arguments 非法 JSON；
- Parser 原因：`empty_response`、`invalid_json`、`invalid_schema`、`unknown_field`、`empty_choice`、`duplicate_value`、`unknown_value`；
- 候选池语义原因：`empty_intersection`、`already_queried_only`、`uninformative_full_pool`。

不重试网络/鉴权/超时等 `_call_llm()` 失败；这些直接保留原 fallback。

重试消息必须符合 OpenAI Tool Calling 合同：

1. 追加原 assistant message，保留完整 `tool_calls`；
2. 对被检查的调用追加 `role="tool"`、匹配的 `tool_call_id` 和简短机器可读错误；
3. 明确要求重新调用同一 tool，并给出当前语义约束，例如“至少覆盖一个未查询候选且不能覆盖全部剩余池”。

重试消息仅存在于 `_llm_mean_shift()` 局部变量，不进入 trajectory 或后续 prompt。

### R5: Deterministic Tool Selection

- 只接受 `propose_sparse_subspace` 的第一个调用；额外调用记为首轮失败并请求只返回一个调用。
- 无匹配调用时，不退回解析任意 prose；兼容纯文本只用于没有 Tool Calling 能力的显式旧路径，不用于本任务的强制 Chem tool 模式。
- 最终保存的 `raw_response` 是被实际解析的 tool arguments 字符串；保持现有 `_extract_thinking(raw_response)` 行为。

### R6: Telemetry & Provenance

`guidance_artifacts` 每步新增：

- `react_retried: bool`；
- `react_first_reason: str | None`；
- `llm_attempts: int`（1 或 2）；
- `tool_call_id: str | None`。

字段写入 `guidance_artifacts`，不写入 trajectory。最终 `parser_reason` 仍表示最终结果：`accepted` 或最终 fallback 原因。

实验 config / `model_config` 同步记录：

- `temperature: 0.2`；
- `response_mode: "tool_call_react"`；
- `max_react_retries: 1`。

### R7: Package / Submission Parity

以下镜像必须保持行为一致：

- `packages/bo-core/bo_core/llm_client.py` ↔ `Compitetion/submission/code/bo_core/llm_client.py`；
- `packages/bo-core/bo_core/optimization/lgbo.py` ↔ submission mirror；
- `packages/bo-core/bo_core/optimization/chem_lgbo.py` ↔ submission mirror。

## Acceptance Criteria

- [ ] 合法 tool call 在 `content=""` 时仍被视为成功并完成解析。
- [ ] `unknown_value` 首轮失败后，第二轮合法调用被接受。
- [ ] `already_queried_only` 首轮失败后，第二轮覆盖部分未查询池的调用被接受。
- [ ] `uninformative_full_pool` 同样可纠错；连续两轮失败则只记录一次最终 fallback。
- [ ] tool retry 回执携带正确 `tool_call_id`，且重试消息不进入 trajectory/下一 step prompt。
- [ ] accepted 与 fallback artifact 均含完整 ReAct telemetry。
- [ ] `temperature=0.2` 和 response mode 出现在实验 provenance 中，不再硬编码为 `0.0`。
- [ ] 基类纯文本 LGBO 和其他 `DeepSeekClient.chat()` 调用保持默认 `temperature=0.0`。
- [ ] package 与 submission 镜像通过等价行为测试。
- [ ] 现有 bo-core 测试、覆盖率、Ruff 与 mypy 通过。
- [ ] 至少用一个真实保存状态完成 Tool Call + ReAct smoke test；报告结构成功率、语义成功率、重试自愈率和最终 fallback 率，不能只报告“返回了 tool call”。

## Non-Goals

- 不把 option 全量 enum 注入 tool schema。
- 不增加超过 1 次重试。
- 不改变 `+1σ` mean shift、EI、counterfactual 或候选选择算法。
- 不把 LLM 建议变成硬过滤。
- 不声称 `temperature=0.2` 提升优化收益；本任务只验证协议正确性和 fallback 自愈。
