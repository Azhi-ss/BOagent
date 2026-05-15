from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from llm_client import DeepSeekClient

ChatState = Literal[
    "idle",
    "awaiting_data_choice",
    "awaiting_demo_confirm",
    "awaiting_goal_confirm",
    "ready_to_run",
    "running_bo",
    "reporting",
]

ActionType = Literal[
    "none",
    "select_demo_data",
    "confirm_demo_data",
    "confirm_goal",
    "create_bo_session",
    "run_demo_bo",
    "run_next_bo_step",
    "run_bo_step",
    "explain_result",
]

KNOWN_STATES = {
    "idle",
    "awaiting_data_choice",
    "awaiting_demo_confirm",
    "awaiting_goal_confirm",
    "ready_to_run",
    "running_bo",
    "reporting",
}

KNOWN_ACTIONS = {
    "none",
    "select_demo_data",
    "confirm_demo_data",
    "confirm_goal",
    "create_bo_session",
    "run_demo_bo",
    "run_next_bo_step",
    "run_bo_step",
    "explain_result",
}


@dataclass
class ChatAgentAction:
    type: ActionType
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatAgentDecision:
    intent: str
    next_state: ChatState
    action: ChatAgentAction
    assistant_message: str
    ui_hints: list[str] = field(default_factory=list)


@dataclass
class ChatAgentConversation:
    conversation_id: str
    state: ChatState = "idle"
    data_source: str | None = None
    data_source_confirmed: bool = False
    demo_disclaimer_confirmed: bool = False
    goal_confirmed: bool = False
    user_confirmed_run: bool = False
    session_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def state_payload(self) -> dict[str, Any]:
        allowed_actions = allowed_actions_for(self)
        return {
            "state": self.state,
            "data_source": self.data_source,
            "data_source_confirmed": self.data_source_confirmed,
            "demo_disclaimer_confirmed": self.demo_disclaimer_confirmed,
            "goal_confirmed": self.goal_confirmed,
            "user_confirmed_run": self.user_confirmed_run,
            "session_id": self.session_id,
            "allowed_actions": sorted(allowed_actions),
            "forbidden_actions": sorted(KNOWN_ACTIONS - allowed_actions),
        }


def allowed_actions_for(conversation: ChatAgentConversation) -> set[str]:
    if conversation.state in {"idle", "awaiting_data_choice"}:
        return {"none", "select_demo_data", "run_demo_bo", "run_bo_step"}
    if conversation.state == "awaiting_demo_confirm":
        return {"none", "confirm_demo_data"}
    if conversation.state == "awaiting_goal_confirm":
        return {"none", "confirm_goal"}
    if conversation.state == "ready_to_run":
        return {"none", "run_bo_step", "run_next_bo_step", "explain_result"}
    if conversation.state in {"running_bo", "reporting"}:
        return {"none", "run_bo_step", "run_next_bo_step", "explain_result"}
    return {"none"}


def safe_next_states_for(
    conversation: ChatAgentConversation,
    action_type: str,
) -> set[str]:
    state = conversation.state
    if state in {"idle", "awaiting_data_choice"}:
        if action_type == "none":
            return {state, "awaiting_data_choice"}
        if action_type == "select_demo_data":
            return {"awaiting_demo_confirm"}
        if action_type in {"run_demo_bo", "run_bo_step"}:
            return {"running_bo"}
    if state == "awaiting_demo_confirm":
        if action_type == "none":
            return {state}
        if action_type == "confirm_demo_data":
            return {"awaiting_goal_confirm"}
    if state == "awaiting_goal_confirm":
        if action_type == "none":
            return {state}
        if action_type == "confirm_goal":
            return {"ready_to_run"}
    if state == "ready_to_run":
        if action_type == "none":
            return {state}
        if action_type == "explain_result":
            return {state, "reporting"}
        if action_type in {"run_bo_step", "run_next_bo_step"}:
            return {"running_bo", "reporting"}
    if state in {"running_bo", "reporting"}:
        if action_type in {"none", "run_bo_step", "run_next_bo_step", "explain_result"}:
            return {state, "running_bo", "reporting"}
    return {state}


def infer_next_state_for_action(
    conversation: ChatAgentConversation,
    action_type: str,
) -> ChatState:
    if action_type == "select_demo_data":
        return "awaiting_demo_confirm"
    if action_type == "confirm_demo_data":
        return "awaiting_goal_confirm"
    if action_type == "confirm_goal":
        return "ready_to_run"
    if action_type in {"run_bo_step", "run_demo_bo", "run_next_bo_step"}:
        return "running_bo"
    if action_type == "explain_result" and conversation.session_id:
        return "reporting"
    if conversation.state == "idle" and not conversation.data_source_confirmed:
        return "awaiting_data_choice"
    return conversation.state


def sanitize_decision_for_conversation(
    conversation: ChatAgentConversation,
    decision: ChatAgentDecision,
    language: str,
) -> ChatAgentDecision:
    allowed_actions = allowed_actions_for(conversation)
    if decision.action.type not in allowed_actions:
        return blocked_decision_for_state(
            language,
            f"unsafe_action_{decision.action.type}",
            conversation.state,
        )
    if (
        decision.action.type == "select_demo_data"
        and decision.action.args.get("source") != "demo_pvk"
    ):
        return unsupported_data_source_decision(language)
    if (
        conversation.state in {"idle", "awaiting_data_choice"}
        and decision.action.type == "none"
        and not conversation.data_source_confirmed
    ):
        decision.next_state = "awaiting_data_choice"
        decision.ui_hints = list(
            dict.fromkeys([*decision.ui_hints, "show_demo_button", "show_upload_disabled"])
        )
    if decision.next_state not in safe_next_states_for(conversation, decision.action.type):
        return blocked_decision_for_state(
            language,
            f"unsafe_state_{decision.next_state}",
            conversation.state,
        )
    return decision


def route_explicit_user_action(
    conversation: ChatAgentConversation,
    decision: ChatAgentDecision,
    message: str,
) -> ChatAgentDecision:
    normalized = message.strip().lower()
    if (
        decision.action.type in {"none", "select_demo_data"}
        and not conversation.data_source_confirmed
        and _mentions_demo_reference(normalized)
        and _mentions_run_intent(normalized)
    ):
        return ChatAgentDecision(
            intent=decision.intent,
            next_state="running_bo",
            action=ChatAgentAction(
                type="run_demo_bo",
                args={"source": "demo_pvk", "goal": "band_alignment_eta"},
            ),
            assistant_message=decision.assistant_message,
            ui_hints=decision.ui_hints,
        )
    if (
        decision.action.type == "none"
        and conversation.session_id
        and _mentions_run_intent(normalized)
    ):
        return ChatAgentDecision(
            intent=decision.intent,
            next_state="running_bo",
            action=ChatAgentAction(type="run_next_bo_step"),
            assistant_message=decision.assistant_message,
            ui_hints=decision.ui_hints,
        )
    return decision


def _mentions_demo_reference(message: str) -> bool:
    return any(term in message for term in ("demo", "演示", "内置", "reference", "参考"))


def _mentions_run_intent(message: str) -> bool:
    return any(term in message for term in ("开始", "直接", "运行", "跑", "继续", "下一轮", "run", "start", "next"))


def can_request_bo_run(conversation: ChatAgentConversation) -> bool:
    return (
        conversation.data_source == "demo_pvk"
        and conversation.data_source_confirmed
        and conversation.demo_disclaimer_confirmed
        and conversation.goal_confirmed
    )


def can_accept_demo_run_request(
    conversation: ChatAgentConversation,
    decision: ChatAgentDecision,
) -> bool:
    if conversation.session_id or decision.action.type not in {"run_bo_step", "run_demo_bo"}:
        return False
    args = decision.action.args
    return (
        args.get("source") == "demo_pvk"
        and args.get("goal") in {"band_alignment_eta", "band_alignment", "eta"}
    )


def apply_demo_run_consent(conversation: ChatAgentConversation) -> None:
    conversation.data_source = "demo_pvk"
    conversation.data_source_confirmed = True
    conversation.demo_disclaimer_confirmed = True
    conversation.goal_confirmed = True
    conversation.user_confirmed_run = False


def blocked_run_decision(language: str) -> ChatAgentDecision:
    return fallback_decision(language, "bo_prerequisites_missing")


def blocked_decision_for_state(
    language: str,
    reason: str,
    state: ChatState,
) -> ChatAgentDecision:
    decision = fallback_decision(language, reason)
    decision.next_state = "awaiting_data_choice" if state == "idle" else state
    return decision


def unsupported_data_source_decision(language: str) -> ChatAgentDecision:
    message = (
        "第一版只支持内置 PVK demo 数据。请先选择内置 demo 数据；上传数据流程会在后续版本开放。"
        if language == "zh"
        else "This first version only supports the built-in PVK demo data. Please choose the built-in demo data; upload support will come later."
    )
    return ChatAgentDecision(
        intent="blocked:unsupported_data_source",
        next_state="awaiting_data_choice",
        action=ChatAgentAction(type="none"),
        assistant_message=message,
        ui_hints=["show_demo_button", "show_upload_disabled"],
    )


def apply_allowed_state_update(
    conversation: ChatAgentConversation,
    decision: ChatAgentDecision,
) -> None:
    action_type = decision.action.type
    if action_type == "select_demo_data":
        source = decision.action.args.get("source")
        if source == "demo_pvk":
            conversation.data_source = "demo_pvk"
            conversation.data_source_confirmed = True
            conversation.demo_disclaimer_confirmed = False
            conversation.goal_confirmed = False
            conversation.user_confirmed_run = False
            conversation.session_id = None
    elif action_type == "confirm_demo_data":
        conversation.demo_disclaimer_confirmed = True
        conversation.user_confirmed_run = False
    elif action_type == "confirm_goal":
        conversation.goal_confirmed = True
        conversation.user_confirmed_run = False
    elif action_type in {"run_bo_step", "run_demo_bo", "run_next_bo_step"}:
        # Task 3 consumes this flag to create/run the BO session after this gate passes.
        conversation.user_confirmed_run = True

    conversation.state = decision.next_state


def build_chat_planner_messages(
    conversation: ChatAgentConversation,
    message: str,
    language: str,
) -> list[dict[str, str]]:
    language_instruction = (
        "Respond in Simplified Chinese." if language == "zh" else "Respond in English."
    )
    system_prompt = (
        "You are a concise PVK Bayesian optimization research agent. "
        "Talk naturally with the user; do not sound like a form wizard. "
        "Never expose internal routing, state names, or policy machinery. "
        "Return only valid JSON with intent, action, assistant_message, and ui_hints. "
        "The backend will infer any internal route from your action. "
        "If no data source is confirmed, ask naturally whether the user wants to use the built-in "
        "PVK demo/reference data or wait for upload support. "
        "If the user clearly says to use the built-in PVK demo/reference data and start or continue, "
        "prefer action run_demo_bo with args source=demo_pvk and goal=band_alignment_eta, while explaining "
        "that it is reference data and not wet-lab validation. "
        "If the user only wants to select the demo data but not start yet, use action select_demo_data. "
        "If the user says continue, yes, confirm, start, or next in context, map that to the most useful available action. "
        "Do not claim BO is running unless a tool action is allowed and requested. "
        "If the user has not uploaded data, never imply user data exists. "
        "If demo data is used, explicitly call it built-in demo/reference data. "
        "Workbook lookup is reference evaluation, not wet-lab validation. "
        "Ask for missing prerequisites conversationally before proposing BO execution. "
        "Keep replies concise and action-oriented."
    )
    user_prompt = {
        "language_instruction": language_instruction,
        "agent_context": public_agent_context(conversation),
        "user_message": message,
        "output_schema": {
            "intent": "string",
            "action": {"type": "one of the available action strings", "args": {}},
            "assistant_message": "string",
            "ui_hints": ["string"],
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def public_agent_context(conversation: ChatAgentConversation) -> dict[str, Any]:
    if conversation.data_source == "demo_pvk":
        data_source = "built-in PVK demo/reference data"
    elif conversation.data_source:
        data_source = conversation.data_source
    else:
        data_source = None
    actions = allowed_actions_for(conversation)
    available_actions = ["none"]
    if "select_demo_data" in actions:
        available_actions.append("select_demo_data")
    if "confirm_demo_data" in actions:
        available_actions.append("confirm_demo_data")
    if "confirm_goal" in actions:
        available_actions.append("confirm_goal")
    if "run_bo_step" in actions and (
        can_request_bo_run(conversation) or not conversation.data_source_confirmed
    ):
        available_actions.append("run_bo_step")
    if "run_demo_bo" in actions and not conversation.data_source_confirmed:
        available_actions.append("run_demo_bo")
    if "run_next_bo_step" in actions and conversation.session_id:
        available_actions.append("run_next_bo_step")
    if "explain_result" in actions and conversation.session_id:
        available_actions.append("explain_result")
    return {
        "data_source": data_source,
        "data_source_confirmed": conversation.data_source_confirmed,
        "demo_reference_acknowledged": conversation.demo_disclaimer_confirmed,
        "goal": "band_alignment, maximize eta/PCE" if conversation.goal_confirmed else None,
        "has_bo_session": bool(conversation.session_id),
        "bo_can_run_now": can_request_bo_run(conversation),
        "demo_bo_shortcut": (
            "If the user clearly asks to use the built-in PVK demo/reference data and start, "
            "you may use action run_demo_bo with args source=demo_pvk and goal=band_alignment_eta."
            if not conversation.data_source_confirmed
            else None
        ),
        "available_actions": available_actions,
        "safety_notes": [
            "Built-in demo/reference data is not uploaded user data.",
            "Workbook lookup is not wet-lab validation.",
        ],
    }


def parse_chat_decision(
    content: str,
    conversation: ChatAgentConversation | None = None,
) -> ChatAgentDecision:
    raw = json.loads(content)
    next_state = raw.get("next_state")
    action_raw = raw.get("action") or {}
    if isinstance(action_raw, str):
        action_type = action_raw
        action_args: dict[str, Any] = {}
    elif isinstance(action_raw, dict):
        action_type = action_raw.get("type")
        action_args = dict(action_raw.get("args") or {})
    else:
        action_type = None
        action_args = {}
    if action_type == "select_demo_data" and "source" not in action_args:
        action_args["source"] = "demo_pvk"
    if next_state is None and conversation is not None and action_type in KNOWN_ACTIONS:
        next_state = infer_next_state_for_action(conversation, action_type)
    if next_state not in KNOWN_STATES:
        raise ValueError(f"Unknown next_state: {next_state}")
    if action_type not in KNOWN_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type}")
    assistant_message = str(raw.get("assistant_message") or "").strip()
    if not assistant_message:
        raise ValueError("assistant_message is required")
    ui_hints = raw.get("ui_hints") or []
    if not isinstance(ui_hints, list):
        ui_hints = []
    return ChatAgentDecision(
        intent=str(raw.get("intent") or "other"),
        next_state=next_state,
        action=ChatAgentAction(
            type=action_type,
            args=action_args,
        ),
        assistant_message=assistant_message,
        ui_hints=[str(item) for item in ui_hints],
    )


def clean_fallback_reason(reason: str) -> str:
    http_match = re.search(r"\bHTTP\s+(\d{3})\b", str(reason))
    if http_match:
        return f"http_{http_match.group(1)}"
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(reason or "unknown")).strip("_")
    return (cleaned.lower() or "unknown")[:40]


def fallback_decision(language: str, reason: str) -> ChatAgentDecision:
    message = (
        "我可以继续帮你规划 PVK 贝叶斯优化，但现在不会运行 BO。请先选择使用内置 demo 数据，或之后上传你的数据。"
        if language == "zh"
        else "I can help plan PVK Bayesian optimization, but I will not run BO yet. Please choose built-in demo data or upload data later."
    )
    return ChatAgentDecision(
        intent=f"fallback:{clean_fallback_reason(reason)}",
        next_state="awaiting_data_choice",
        action=ChatAgentAction(type="none"),
        assistant_message=message,
        ui_hints=["show_demo_button", "show_upload_disabled"],
    )


class ChatAgentPlanner:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client or DeepSeekClient.from_env()

    def plan(
        self,
        conversation: ChatAgentConversation,
        message: str,
        language: str,
    ) -> ChatAgentDecision:
        result = self.client.chat(
            build_chat_planner_messages(conversation, message, language),
            max_tokens=900,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if result.status != "success":
            return fallback_decision(language, result.error or result.status)
        try:
            decision = parse_chat_decision(result.content, conversation)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return fallback_decision(language, type(exc).__name__)
        return sanitize_decision_for_conversation(conversation, decision, language)

    def summarize_bo_result(
        self,
        artifacts: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        language: str,
    ) -> str:
        fallback = format_bo_result_summary(artifacts, language)
        if not isinstance(self.client, DeepSeekClient):
            return fallback
        result = self.client.chat(
            build_bo_result_messages(artifacts, tool_calls, language),
            max_tokens=700,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if result.status != "success" or not result.content.strip():
            return fallback
        return result.content.strip()


def format_bo_result_summary(artifacts: dict[str, Any], language: str) -> str:
    bo_step = artifacts.get("bo_step") if isinstance(artifacts.get("bo_step"), dict) else {}
    best_score = bo_step.get("best_score")
    best_result = bo_step.get("best_result") if isinstance(bo_step.get("best_result"), dict) else {}
    config = best_result.get("config") if isinstance(best_result.get("config"), dict) else {}
    selected = bo_step.get("selected_candidate")
    selected_label = ""
    if isinstance(selected, dict):
        selected_label = str(selected.get("candidate_id") or selected.get("id") or "")
    if language == "en":
        parts = [
            f"BO step finished. Current best reference eta/PCE score is {best_score}.",
        ]
        if config:
            params = ", ".join(f"{key}={value}" for key, value in config.items())
            parts.append(f"Best parameters: {params}.")
        if selected_label:
            parts.append(f"Selected candidate: {selected_label}.")
        parts.append("This is workbook/reference lookup, not wet-lab validation. Continue to the next BO step if you want another proposal.")
        return " ".join(parts)
    parts = [f"这一轮 BO 已完成。当前最优 reference eta/PCE 分数是 {best_score}。"]
    if config:
        params = "，".join(f"{key}={value}" for key, value in config.items())
        parts.append(f"最佳参数：{params}。")
    if selected_label:
        parts.append(f"本轮选中的候选点：{selected_label}。")
    parts.append("这来自 workbook/reference lookup，不是湿实验验证。你可以继续下一轮 BO。")
    return "".join(parts)


def build_bo_result_messages(
    artifacts: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    language: str,
) -> list[dict[str, str]]:
    language_instruction = (
        "Respond in Simplified Chinese." if language == "zh" else "Respond in English."
    )
    system_prompt = (
        "You are a concise PVK Bayesian optimization research agent. "
        "Summarize the BO tool result as a natural research update. "
        "Mention the best score, best parameters if present, what the tool chain did, "
        "and one clear next-step suggestion. "
        "Do not expose internal state names. Do not claim wet-lab validation."
    )
    user_payload = {
        "language_instruction": language_instruction,
        "bo_result": artifacts.get("bo_step", {}),
        "data_boundary": artifacts.get("data_boundary"),
        "tool_calls": [call.get("name") for call in tool_calls],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


class ChatAgentStore:
    def __init__(self, planner_client: Any | None = None) -> None:
        self.planner = ChatAgentPlanner(client=planner_client)
        self._conversations: dict[str, ChatAgentConversation] = {}

    def get_or_create(self, conversation_id: str | None) -> ChatAgentConversation:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        conversation = ChatAgentConversation(conversation_id=f"chat_{uuid4().hex[:12]}")
        self._conversations[conversation.conversation_id] = conversation
        return conversation
