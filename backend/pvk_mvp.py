from __future__ import annotations

from typing import Any


CURVE_BOUNDARY = "session best-so-far only, not benchmark"


def build_passivation_target(session_or_artifacts: dict | None = None) -> dict:
    """Build the target card for the PVK passivation MVP."""
    current_best = _extract_current_best(session_or_artifacts)
    champion_threshold = {
        "metric": "PCE",
        "operator": ">=",
        "value": round(max(current_best + 0.2, 26.5), 2) if current_best else 26.5,
        "unit": "%",
        "note": "Demo 门槛用于展示 champion 判定，不代表已验证实验阈值。",
    }

    return {
        "title": "钙钛矿钝化配方优化 Target",
        "objective": (
            "在正常带隙钙钛矿太阳能电池中，通过多智能体 BO/LLM-BO 工作流筛选钝化策略，"
            "优先提升 PCE，同时监控 Voc、FF、稳定性与工艺可重复性。"
        ),
        "passivators": {
            "3MTPAI": {
                "role": "芳香铵盐候选，用作界面缺陷钝化与疏水性调节的探索项。",
                "risk": "中风险",
                "evidence_level": "demo 先验 + 小样本趋势，需要真实实验复核。",
            },
            "PDAI2": {
                "role": "二铵盐候选，偏向晶界/界面协同钝化与离子迁移抑制假设。",
                "risk": "中低风险",
                "evidence_level": "demo 数据中有间接支持，可作为稳健候选。",
            },
            "EDAI2": {
                "role": "短链二铵盐基线候选，适合作为 exploit/control 的主锚点。",
                "risk": "低到中风险",
                "evidence_level": "demo 数据支持度最高，但仍不是真实 benchmark 结论。",
            },
            "PipDI": {
                "role": "高探索性环状二铵盐候选，仅用于提出待验证假设。",
                "risk": "高风险",
                "evidence_level": "无真实样本；只能作为 demo-only 探索，不可当作已验证配方。",
            },
        },
        "recommended_strategy": (
            "先以 EDAI2/PDAI2 建立低风险 exploitation 批次，再用 3MTPAI 做结构多样性扩展；"
            "PipDI 仅在安全小剂量与对照充分时进入探索批次。"
        ),
        "data_boundary": (
            "当前 passivation ratio 是 strategy/combination 层面的策略变量，"
            "不是真实 molar ratio BO；所有输出只服务 MVP 展示。"
        ),
        "champion_threshold": champion_threshold,
    }


def compute_bo_curve(observed_fvals: list[float] | None = None) -> dict:
    """Return only the session best-so-far curve, without fabricated baselines."""
    if not observed_fvals:
        pvk_points = _points([22.0, 23.1, 24.0, 24.8, 25.5, 26.0, 26.3])
        pvk_boundary = "demo walkthrough values, not benchmark"
    else:
        pvk_points = _points(_best_so_far(observed_fvals))
        pvk_boundary = "observed best-so-far from supplied fvals"

    return {
        "curve_boundary": CURVE_BOUNDARY,
        "series": {
            "pvk_bo": {
                "label": "PVK Agent BO/LLM-BO",
                "points": pvk_points,
                "boundary": pvk_boundary,
            },
        },
    }


def generate_screening_analysis(target: dict, session: dict | None = None) -> dict:
    """Generate Step II screening reasoning for the MVP."""
    observed = session.get("observed_fvals", []) if isinstance(session, dict) else []
    best_note = ""
    if observed:
        best_note = f"当前观测 best-so-far PCE 为 {max(observed):.2f}%，仅作会话内参考。"

    rankings = [
        {
            "rank": 1,
            "passivator": "EDAI2",
            "score": 0.88,
            "risk": target["passivators"]["EDAI2"]["risk"],
            "rationale": "作为短链二铵盐锚点，demo 小样本支持度最高，适合作为首轮 exploit/control。",
        },
        {
            "rank": 2,
            "passivator": "PDAI2",
            "score": 0.8,
            "risk": target["passivators"]["PDAI2"]["risk"],
            "rationale": "二铵盐结构与晶界钝化假设相容，适合与 EDAI2 形成组合策略。",
        },
        {
            "rank": 3,
            "passivator": "3MTPAI",
            "score": 0.68,
            "risk": target["passivators"]["3MTPAI"]["risk"],
            "rationale": "提供芳香铵盐结构多样性，但需要用对照拆分疏水性与缺陷钝化贡献。",
        },
        {
            "rank": 4,
            "passivator": "PipDI",
            "score": 0.35,
            "risk": target["passivators"]["PipDI"]["risk"],
            "rationale": "PipDI 无真实样本，只能保留为高风险探索假设，不能进入 champion 叙事。",
        },
    ]

    small_sample_induction = (
        "小样本归纳：先把 EDAI2/PDAI2 视作低风险结构锚点，再用 3MTPAI 做可解释扩展；"
        "PipDI 因缺少真实样本，只能作为边界清晰的探索项。"
    )
    language_level_explanation = (
        "语言层解释：agent 将 passivator 的官能团、链长、界面作用和风险词汇转成排序先验，"
        "但这些文本先验不能替代真实实验。"
    )

    return {
        "phase": "Screening",
        "prior_knowledge": [
            "优先选择已有 demo 支持的二铵盐类候选，降低首轮实验风险。",
            "把稳定性、Voc/FF 协同变化作为 PCE 之外的 critic 维度。",
            "passivation ratio 当前只表达 strategy/combination，不是真实 molar ratio BO。",
        ],
        "small_sample_induction": small_sample_induction,
        "小样本归纳": small_sample_induction,
        "language_level_explanation": language_level_explanation,
        "语言层解释": language_level_explanation,
        "candidate_rankings": rankings,
        "data_boundary": target["data_boundary"],
        "session_note": best_note or "未提供会话观测值，使用 demo screening 解释。",
    }


def handle_mvp_chat_turn(
    message: str, session: dict | None = None, language: str = "zh"
) -> dict:
    """Template a local, no-network MVP chat response."""
    phase = _infer_phase(message)
    target = build_passivation_target(session)
    tool_calls: list[dict[str, Any]] = [
        {"name": "build_passivation_target", "arguments": {"session_or_artifacts": "session"}}
    ]
    artifacts: dict[str, Any] = {"target": target}

    if phase == "Screening":
        screening = generate_screening_analysis(target, session)
        artifacts["screening_analysis"] = screening
        tool_calls.append(
            {
                "name": "generate_screening_analysis",
                "arguments": {"target": "artifacts.target", "session": "session"},
            }
        )
        assistant_message = (
            "Step II Screening 已完成：我会优先把 EDAI2/PDAI2 作为低风险锚点，"
            "3MTPAI 用于结构多样性探索。PipDI 无真实样本，因此标记为高风险，"
            "只能作为 demo-only 假设。当前 passivation ratio 是 strategy/combination，"
            "不是真实 molar ratio BO。"
        )
    elif phase == "Optimization":
        observed_fvals = _session_observed_fvals(session)
        curve = compute_bo_curve(observed_fvals)
        artifacts["bo_curve"] = curve
        tool_calls.append(
            {"name": "compute_bo_curve", "arguments": {"observed_fvals": observed_fvals}}
        )
        assistant_message = (
            "Step III Optimization 已生成 demo 曲线：PVK Agent BO/LLM-BO 使用会话观测的 "
            "best-so-far 或合成演示点；曲线边界是 session best-so-far only, not benchmark，"
            "不能解读为真实算法性能。"
        )
    else:
        assistant_message = (
            "Step I Initialization 已建立 Target：目标是提升 PCE，同时约束 Voc、FF、稳定性和复现性。"
            "候选包含 3MTPAI、PDAI2、EDAI2、PipDI，其中 PipDI 因无真实样本被标注为高风险。"
        )

    if language != "zh":
        assistant_message = (
            assistant_message
            + " Summary: this MVP is local and synthetic; no network calls or benchmark claims are made."
        )

    return {
        "assistant_message": assistant_message,
        "phase": phase,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
    }


def _extract_current_best(session_or_artifacts: dict | None) -> float:
    if not isinstance(session_or_artifacts, dict):
        return 0.0

    best_result = session_or_artifacts.get("best_result")
    if isinstance(best_result, dict):
        value = best_result.get("score") or best_result.get("pce") or best_result.get("PCE")
        return _safe_float(value)

    summary = session_or_artifacts.get("summary")
    if isinstance(summary, dict):
        return _safe_float(summary.get("best_score") or summary.get("best_pce"))

    observed = session_or_artifacts.get("observed_fvals")
    if isinstance(observed, list) and observed:
        return max(_safe_float(value) for value in observed)

    return 0.0


def _session_observed_fvals(session: dict | None) -> list[float] | None:
    if isinstance(session, dict) and isinstance(session.get("observed_fvals"), list):
        return session["observed_fvals"]
    return None


def _infer_phase(message: str) -> str:
    normalized = message.lower()
    if any(token in normalized for token in ("initial", "初始化", "开始任务", "target")):
        return "Initialization"
    if any(token in normalized for token in ("screen", "筛选", "screening", "pipdi")):
        return "Screening"
    if any(token in normalized for token in ("optimiz", "优化", "curve", "曲线", "bo")):
        return "Optimization"
    return "Initialization"


def _best_so_far(values: list[float]) -> list[float]:
    best_values: list[float] = []
    current_best: float | None = None
    for value in values:
        numeric_value = _safe_float(value)
        current_best = numeric_value if current_best is None else max(current_best, numeric_value)
        best_values.append(current_best)
    return best_values


def _points(values: list[float]) -> list[dict[str, float | int]]:
    return [
        {"iteration": index + 1, "pce": round(float(value), 3)}
        for index, value in enumerate(values)
    ]


def _series_values(seed_values: list[float], target_length: int) -> list[float]:
    if target_length <= len(seed_values):
        return seed_values[:target_length]

    values = list(seed_values)
    while len(values) < target_length:
        values.append(round(values[-1] + 0.08, 3))
    return values


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

