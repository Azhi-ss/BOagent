"""全矩阵对比：4 个模型 × 有详细说明 Tool Calling 效果测试"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BO_CORE_PATH = Path("/home/dministrator/project/BOagent/packages/bo-core")
AUTO_RESEARCH_PATH = Path("/home/dministrator/project/BOagent/Compitetion/auto_research")
if str(BO_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(BO_CORE_PATH))
if str(AUTO_RESEARCH_PATH) not in sys.path:
    sys.path.insert(0, str(AUTO_RESEARCH_PATH))

import os

from bo_core.llm_client import load_env_file
from components.cake import BASE_KERNELS, OPERATORS

load_env_file()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://newapi.ai-modeling.top/v1")

if not API_KEY or API_KEY == "your_deepseek_api_key_here":
    raise SystemExit(
        "DEEPSEEK_API_KEY missing — set it in the repo-root .env before running."
    )

MODELS = [
    "fxb-deepseek-v4-flash",
    "fxb-minimax-2.7",
    "sensenova-6.7-flash-lite",
    "glm-5-2-260617",
]

TOOL_DETAILED = {
    "type": "function",
    "function": {
        "name": "propose_evolved_kernel",
        "description": (
            "Submit an evolved Gaussian Process (GP) composite kernel expression "
            "for chemical reaction Bayesian optimization. "
            "Use this tool to propose a new, higher-fitness kernel by combining "
            "or mutating parent kernels based on observed reaction yield data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kernel_expression": {
                    "type": "string",
                    "description": (
                        "A valid mathematical kernel expression string. "
                        "MUST ONLY contain allowed base kernels: "
                        "['SE', 'PER', 'LIN', 'RQ', 'M1', 'M3', 'M5'] "
                        "and allowed operators: ['+', '*']. "
                        "Use parentheses for grouping. "
                        "Valid examples: 'LIN * PER', '(SE + PER) * RQ', 'M5 * LIN'."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "A detailed physical and chemical domain reasoning explaining "
                        "why this kernel combination captures the trends in reaction "
                        "components (base strength, ligand steric/electronic effects, "
                        "additive variations, and catalyst substituent periodicity)."
                    ),
                },
            },
            "required": ["kernel_expression", "reasoning"],
        },
    },
}


def call_api(model: str) -> dict:
    """Send Tool Calling request, return result dict."""
    system_prompt = (
        "You are an expert material science & AI agent specializing in "
        "Gaussian process kernel design for chemical reaction optimization.\n"
        f"Allowed base kernels: {', '.join(BASE_KERNELS)}\n"
        f"Allowed operators: {', '.join(OPERATORS)}\n"
        "Observe the reaction data and call 'propose_evolved_kernel'."
    )
    user_prompt = (
        "Parent Kernel 1: 'LIN * PER' (fitness: 0.882)\n"
        "Parent Kernel 2: 'SE + M5' (fitness: 0.741)\n"
        "Observations show high yield sensitivity to base type "
        "(MTBD vs BTMG) and catalyst substituent periodicity.\n"
        "Please propose an evolved composite kernel."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": False,
        "tools": [TOOL_DETAILED],
        "tool_choice": {
            "type": "function",
            "function": {"name": "propose_evolved_kernel"},
        },
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        elapsed = time.time() - t0
        status = resp.status_code
        data = resp.json()

        if "error" in data:
            return {
                "status": status,
                "ok": False,
                "error": data["error"].get("message", str(data["error"])),
                "elapsed": elapsed,
            }

        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
            return {
                "status": status,
                "ok": True,
                "kernel": args.get("kernel_expression", ""),
                "reasoning": args.get("reasoning", ""),
                "elapsed": elapsed,
            }

        return {
            "status": status,
            "ok": False,
            "error": f"No tool_calls. Content: {msg.get('content', '')[:120]}",
            "elapsed": elapsed,
        }
    except Exception as exc:
        return {"status": 0, "ok": False, "error": str(exc), "elapsed": time.time() - t0}


def main():
    print("=" * 80)
    print(" 🔬 4 模型 Tool Calling 全面对比测试")
    print(f" API: {BASE_URL}")
    print(f" 模型: {', '.join(MODELS)}")
    print("=" * 80)

    results = {}

    for model in MODELS:
        print(f"\n{'━' * 80}")
        print(f" 🤖 {model}")
        print(f"{'━' * 80}")
        r = call_api(model)
        results[model] = r

        if r["ok"]:
            has_parens = "✅" if "(" in r["kernel"] else "❌"
            print(f"   📡 HTTP {r['status']} — ✅ 成功 ({r['elapsed']:.1f}s)")
            print(f"   核: {r['kernel']!r}  (括号: {has_parens})")
            print(f"   推理 ({len(r['reasoning'])} 字):")
            print(f"   {r['reasoning'][:200]}...")
        else:
            print(f"   📡 HTTP {r['status']} — ❌ 失败 ({r['elapsed']:.1f}s)")
            print(f"   错误: {r['error']}")

    # ============================================================
    # 汇总表
    # ============================================================
    print(f"\n\n{'=' * 80}")
    print(" 📊 全矩阵汇总表")
    print(f"{'=' * 80}")
    print(f"{'模型':<28} | {'状态':<6} | {'核表达式':<30} | {'推理长度':>8} | {'括号':>4} | {'耗时':>6}")
    print("-" * 100)

    for model in MODELS:
        r = results[model]
        if r["ok"]:
            ke = r["kernel"]
            rlen = len(r["reasoning"])
            paren = "✅" if "(" in ke else "❌"
            print(f"{model:<28} | {'✅ OK':<6} | {ke:<30} | {rlen:>6} 字 | {paren:>4} | {r['elapsed']:>5.1f}s")
        else:
            err_short = r["error"][:35]
            print(f"{model:<28} | {'❌':<6} | {err_short:<30} | {'—':>8} | {'—':>4} | {r['elapsed']:>5.1f}s")


if __name__ == "__main__":
    main()
