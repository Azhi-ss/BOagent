import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  getArtifacts,
  getBoCurve,
  getSessionId,
  getTasks,
  runNextStep,
  sendAgentChatMessage,
} from "./lib/api";
import type {
  BoCurvePoint,
  BoCurveResponse,
  ChatAgentResponse,
  ChatAgentState,
  ChatMessage,
  ChatResponse,
  JsonMap,
  OptimizationSession,
  Task,
} from "./types";

const STARTER_MESSAGES: ChatMessage[] = [
  {
    role: "agent",
    content:
      "我是 PVK BO 研究助理。你可以直接告诉我想优化什么；当前版本可用内置 PVK reference 数据演示一轮 BO，上传自有数据会在后续开放。",
  },
];

const DEFAULT_PROMPT = "";

function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [session, setSession] = useState<OptimizationSession | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [agentState, setAgentState] = useState<ChatAgentState>("awaiting_data_choice");
  const [messages, setMessages] = useState<ChatMessage[]>(STARTER_MESSAGES);
  const [input, setInput] = useState(DEFAULT_PROMPT);
  const [curve, setCurve] = useState<BoCurvePoint[]>([]);
  const [evidence, setEvidence] = useState<JsonMap>({});
  const [toolCalls, setToolCalls] = useState<JsonMap[]>([]);
  const [phase, setPhase] = useState("ready");
  const [error, setError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isRunningStep, setIsRunningStep] = useState(false);

  const sessionId = getSessionId(session);
  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId || task.task_id === selectedTaskId),
    [selectedTaskId, tasks],
  );
  const latestBest = getLatestBest(curve);
  const canRunBoStep = Boolean(
    sessionId && (agentState === "ready_to_run" || agentState === "reporting"),
  );

  useEffect(() => {
    let isMounted = true;

    async function loadTasks() {
      try {
        const nextTasks = await getTasks();
        if (!isMounted) {
          return;
        }
        setTasks(nextTasks);
        const preferredTask =
          nextTasks.find((task) => (task.task_id || task.id) === "band_alignment") ||
          nextTasks[0];
        setSelectedTaskId(preferredTask?.task_id || preferredTask?.id || "");
      } catch (loadError) {
        if (isMounted) {
          setError(loadError instanceof Error ? loadError.message : "任务列表加载失败");
        }
      }
    }

    void loadTasks();

    return () => {
      isMounted = false;
    };
  }, []);

  const syncArtifacts = useCallback(async (nextSessionId: string) => {
    const artifactResult = await getArtifacts(nextSessionId);
    if (!Array.isArray(artifactResult)) {
      setEvidence(artifactResult as JsonMap);
    }
  }, []);

  const syncAgentOutputs = useCallback(async (nextSessionId: string) => {
    const [curveResult, artifactResult] = await Promise.allSettled([
      getBoCurve(nextSessionId),
      getArtifacts(nextSessionId),
    ]);

    if (curveResult.status === "fulfilled") {
      const nextCurve = normalizeBoCurve(curveResult.value);
      setCurve(nextCurve);
    }

    if (artifactResult.status === "fulfilled") {
      if (!Array.isArray(artifactResult.value)) {
        setEvidence(artifactResult.value as JsonMap);
      }
    }
  }, []);

  const submitAgentMessage = useCallback(
    async (rawMessage: string) => {
      const message = rawMessage.trim();
      if (!message || isSending) {
        return;
      }

      const userMessage: ChatMessage = { role: "user", content: message };
      const nextHistory = [...messages, userMessage];
      setMessages(nextHistory);
      setInput("");
      setError(null);
      setIsSending(true);

      try {
        const response = await sendAgentChatMessage({
          conversation_id: conversationId,
          message,
          history: normalizeChatHistory(nextHistory.slice(STARTER_MESSAGES.length)),
        });
        setConversationId(response.conversation_id);
        setAgentState(response.state);
        setMessages([...nextHistory, ...normalizeChatResponse(response)]);
        setPhase(response.session_id ? "BO result" : "conversation");
        setToolCalls(response.tool_calls || []);
        applyResponseArtifacts(response);

        if (response.session_id) {
          const boStep = getJsonMap(response.artifacts?.bo_step);
          const bestResult = getJsonMap(boStep.best_result);
          const selectedCandidate = getJsonMap(boStep.selected_candidate);
          setSession((current) => ({
            ...(current || {}),
            session_id: response.session_id || undefined,
            status: response.state === "reporting" ? "completed" : response.state,
            iteration: asNumber(boStep.current_step) ?? current?.iteration,
            best_result:
              Object.keys(bestResult).length > 0
                ? {
                    score: asNumber(bestResult.score),
                    parameters: getPrimitiveRecord(bestResult.config),
                  }
                : current?.best_result,
            candidate_points:
              Object.keys(selectedCandidate).length > 0
                ? [
                    {
                      id: String(selectedCandidate.candidate_id || selectedCandidate.id || "selected"),
                      parameters: getPrimitiveRecord(selectedCandidate),
                    },
                  ]
                : current?.candidate_points,
          }));
          await syncAgentOutputs(response.session_id);
        }
      } catch (submitError) {
        const messagePrefix = "Agent 对话请求失败";
        const detail = submitError instanceof Error ? submitError.message : "";
        setError(detail ? `${messagePrefix}：${detail}` : messagePrefix);
        setMessages([
          ...nextHistory,
          {
            role: "agent",
            content: "Agent 对话请求失败。请检查后端服务状态后重试。",
          },
        ]);
      } finally {
        setIsSending(false);
      }
    },
    [conversationId, isSending, messages, syncAgentOutputs],
  );

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) {
      return;
    }

    await submitAgentMessage(input);
  };

  const handleRunOptimizationStep = async () => {
    if (isRunningStep || !sessionId || !canRunBoStep) {
      return;
    }

    setError(null);
    setIsRunningStep(true);
    try {
      const nextSession = await runNextStep(sessionId);
      setSession(nextSession);
      setToolCalls(normalizeToolTrace(nextSession.tool_trace || [], nextSession.iteration || 0));
      setEvidence((current) => ({
        ...current,
        bo_step: {
          current_step: nextSession.iteration || asNumber((nextSession as JsonMap).current_step) || 0,
          best_score: nextSession.best_result?.score,
          best_result: nextSession.best_result,
          selected_candidate: nextSession.candidate_points?.[0],
        },
      }));
      await syncArtifacts(sessionId);
      await syncAgentOutputs(sessionId);
      setAgentState("reporting");
      setPhase("BO result");
      setMessages((current) => [
        ...current,
        {
          role: "agent",
          content:
            "我已调用 BO step 工具推进一轮优化，并同步了右侧 BO 曲线、观测摘要和 artifact。",
        },
      ]);
    } catch (stepError) {
      setError(stepError instanceof Error ? stepError.message : "运行 BO step 失败");
    } finally {
      setIsRunningStep(false);
    }
  };

  function applyResponseArtifacts(response: ChatAgentResponse) {
    const responseArtifacts = response.artifacts || {};
    setEvidence(responseArtifacts);
    const curveArtifact = responseArtifacts.bo_curve;

    if (isBoCurveResponse(curveArtifact)) {
      const nextCurve = normalizeBoCurve(curveArtifact);
      setCurve(nextCurve.length > 0 ? nextCurve : curve);
    }
  }

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#f7f7f4] px-4 py-5 text-neutral-950 sm:px-6">
      <div className="mx-auto flex min-h-[calc(100vh-2.5rem)] max-w-6xl flex-col gap-4">
        <header className="rounded-[1.75rem] border border-neutral-200 bg-white px-5 py-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-neutral-500">BOagent</p>
              <h1 className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-neutral-950 sm:text-3xl">
                PVK BO Research Agent
              </h1>
            </div>
            <StatusPill label={sessionId ? "connected" : "conversation"} />
          </div>
        </header>

        <section className="grid min-w-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
          <AgentChatPanel
            agentState={agentState}
            error={error}
            input={input}
            isSending={isSending}
            messages={messages}
            onInputChange={setInput}
            onQuickAction={submitAgentMessage}
            onSubmit={handleSubmit}
          />

          <ArtifactSidebar
            canRunStep={canRunBoStep}
            curve={curve}
            evidence={evidence}
            isRunningStep={isRunningStep}
            latestBest={latestBest}
            onRunStep={handleRunOptimizationStep}
            phase={phase}
            selectedTask={selectedTask}
            session={session}
            toolCalls={toolCalls}
          />
        </section>
      </div>
    </main>
  );
}

function AgentChatPanel({
  agentState,
  error,
  input,
  isSending,
  messages,
  onInputChange,
  onQuickAction,
  onSubmit,
}: {
  agentState: ChatAgentState;
  error: string | null;
  input: string;
  isSending: boolean;
  messages: ChatMessage[];
  onInputChange: (value: string) => void;
  onQuickAction: (message: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const quickAction = getQuickAction(agentState);

  return (
    <section className="flex h-[calc(100vh-9rem)] min-h-[560px] min-w-0 flex-col overflow-hidden rounded-[1.75rem] border border-neutral-200 bg-white p-4 shadow-sm sm:p-5">
      <div className="flex shrink-0 flex-wrap items-start justify-between gap-3 border-b border-neutral-200 pb-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-neutral-500">
            Conversation
          </p>
          <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em] text-neutral-950">
            和研究助理对话
          </h2>
        </div>
        <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-xs font-medium text-neutral-600">
          concise agent
        </span>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto py-5 pr-2">
        {messages.map((message, index) => (
          <ChatBubble key={`${message.role}-${index}-${message.content.slice(0, 18)}`} message={message} />
        ))}
      </div>

      <div className="mb-4 flex shrink-0 flex-wrap items-center gap-2 rounded-2xl border border-neutral-200 bg-neutral-50 p-3">
        {quickAction ? (
          <button
            className="cursor-pointer rounded-full border border-neutral-900 bg-neutral-950 px-4 py-2 text-xs font-semibold text-white transition-colors duration-200 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isSending}
            onClick={() => onQuickAction(quickAction.message)}
            type="button"
          >
            {quickAction.label}
          </button>
        ) : (
          <span className="rounded-full border border-neutral-200 px-4 py-2 text-xs text-neutral-500">
            继续对话
          </span>
        )}
        {agentState === "awaiting_data_choice" ? (
          <span className="rounded-full border border-neutral-200 px-4 py-2 text-xs text-neutral-500">
            上传数据稍后开放
          </span>
        ) : null}
      </div>

      <form className="shrink-0 border-t border-neutral-200 pt-4" onSubmit={onSubmit}>
        <textarea
          className="min-h-28 w-full resize-none rounded-2xl border border-neutral-200 bg-white p-4 text-sm leading-6 text-neutral-900 placeholder:text-neutral-400 transition-colors duration-200 focus:border-neutral-900 focus:outline-none"
          onChange={(event) => onInputChange(event.target.value)}
          placeholder="例如：用内置 reference 数据跑一轮 BO，并解释结果..."
          value={input}
        />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs leading-5 text-neutral-500">
            工具调用会在后台完成；Agent 会用对话总结结果。
          </p>
          <button
            className="cursor-pointer rounded-2xl bg-[#d4af37] px-6 py-3 text-sm font-semibold text-neutral-950 transition-colors duration-200 hover:bg-[#c8a32f] disabled:cursor-not-allowed disabled:opacity-50"
            disabled={isSending || input.trim().length === 0}
            type="submit"
          >
            {isSending ? "思考中..." : "发送"}
          </button>
        </div>
      </form>

      {error ? (
        <div className="mt-4 shrink-0 rounded-2xl border border-red-200 bg-red-50 p-3 text-sm leading-6 text-red-700">
          {error}
        </div>
      ) : null}
    </section>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isAgent = message.role === "agent" || message.role === "assistant";
  return (
    <article
      className={`min-w-0 max-w-[88%] rounded-3xl border p-4 ${
        isAgent
          ? "border-neutral-200 bg-neutral-50"
          : "ml-auto border-neutral-900 bg-neutral-950 text-white"
      }`}
    >
      <p
        className={`text-[10px] font-semibold uppercase tracking-[0.18em] ${
          isAgent ? "text-neutral-500" : "text-neutral-400"
        }`}
      >
        {isAgent ? "Agent" : "You"}
      </p>
      <p
        className={`mt-2 whitespace-pre-wrap break-words text-sm leading-6 ${
          isAgent ? "text-neutral-900" : "text-white"
        }`}
      >
        {message.content}
      </p>
    </article>
  );
}

function ArtifactSidebar({
  canRunStep,
  curve,
  evidence,
  isRunningStep,
  latestBest,
  onRunStep,
  phase,
  selectedTask,
  session,
  toolCalls,
}: {
  canRunStep: boolean;
  curve: BoCurvePoint[];
  evidence: JsonMap;
  isRunningStep: boolean;
  latestBest: number | null;
  onRunStep: () => void;
  phase: string;
  selectedTask: Task | undefined;
  session: OptimizationSession | null;
  toolCalls: JsonMap[];
}) {
  const boStep = getJsonMap(evidence.bo_step);
  const bestResult = getJsonMap(boStep.best_result);
  const bestConfig = getPrimitiveRecord(bestResult.config);
  const selectedCandidate = getJsonMap(boStep.selected_candidate);
  const observedFvals = Array.isArray(boStep.observed_fvals) ? boStep.observed_fvals : [];
  const bestScore = asNumber(boStep.best_score) ?? asNumber(bestResult.score) ?? latestBest;
  const dataBoundary = getJsonMap(evidence.data_boundary);

  return (
    <aside className="space-y-4">
      <ArtifactCard title="当前证据">
        <div className="grid grid-cols-2 gap-3">
          <MiniStat label="best score" value={bestScore == null ? "--" : bestScore.toFixed(4)} />
          <MiniStat label="step" value={asNumber(boStep.current_step) ?? session?.iteration ?? 0} />
        </div>
        <p className="mt-3 text-xs leading-5 text-neutral-500">
          {selectedTask?.title || selectedTask?.name || selectedTask?.id || "band_alignment"} · {phase}
        </p>
        {Object.keys(dataBoundary).length > 0 ? (
          <p className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">
            {String(dataBoundary.notes || "Workbook/reference lookup，不是湿实验验证。")}
          </p>
        ) : null}
      </ArtifactCard>

      <ArtifactCard title="最佳参数">
        {Object.keys(bestConfig).length > 0 ? (
          <dl className="space-y-2">
            {Object.entries(bestConfig).map(([key, value]) => (
              <div className="flex items-center justify-between gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 px-3 py-2" key={key}>
                <dt className="text-xs font-medium text-neutral-500">{key}</dt>
                <dd className="font-mono text-xs text-neutral-950">{String(value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-sm leading-6 text-neutral-500">运行 BO 后会显示 best 参数。</p>
        )}
        {Object.keys(selectedCandidate).length > 0 ? (
          <p className="mt-3 text-xs leading-5 text-neutral-500">
            选中候选：{String(selectedCandidate.candidate_id || selectedCandidate.id || "selected")}
          </p>
        ) : null}
      </ArtifactCard>

      <ArtifactCard title="BO 曲线">
        <CurveSparkline points={curve} />
        {observedFvals.length > 0 ? (
          <p className="mt-2 text-xs leading-5 text-neutral-500">
            observed fvals: {observedFvals.map((item) => String(item)).join(" / ")}
          </p>
        ) : null}
        <button
          className="mt-4 w-full cursor-pointer rounded-2xl border border-neutral-900 bg-neutral-950 px-4 py-3 text-sm font-semibold text-white transition-colors duration-200 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={isRunningStep || !canRunStep}
          onClick={onRunStep}
          type="button"
        >
          {isRunningStep
            ? "Agent 正在调用 BO 工具..."
            : canRunStep
              ? "继续下一轮 BO"
              : "等待 Agent 创建 BO session"}
        </button>
      </ArtifactCard>

      <ArtifactCard title="工具调用">
        {toolCalls.length > 0 ? (
          <details className="group">
            <summary className="cursor-pointer rounded-2xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm font-medium text-neutral-700 transition-colors duration-200 hover:bg-neutral-100">
              查看 {toolCalls.length} 个工具调用
            </summary>
            <div className="mt-3 space-y-2">
              {toolCalls.map((call, index) => (
                <div className="rounded-2xl border border-neutral-200 bg-white p-3" key={`${call.name}-${index}`}>
                  <p className="font-mono text-xs text-neutral-950">{String(call.name || `tool_${index + 1}`)}</p>
                  <p className="mt-1 truncate text-[11px] text-neutral-500">
                    {JSON.stringify(call.arguments || {})}
                  </p>
                </div>
              ))}
            </div>
          </details>
        ) : (
          <p className="text-sm leading-6 text-neutral-500">工具调用会折叠在这里，不抢占主对话。</p>
        )}
      </ArtifactCard>
    </aside>
  );
}

function ArtifactCard({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <section className="rounded-[1.5rem] border border-neutral-200 bg-white p-4 shadow-sm">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-neutral-500">{title}</p>
      {children}
    </section>
  );
}

function MiniStat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-3">
      <p className="text-[10px] font-semibold uppercase text-neutral-500">{label}</p>
      <p className="mt-1 truncate text-sm font-semibold text-neutral-950">{value}</p>
    </div>
  );
}

function CurveSparkline({ points }: { points: BoCurvePoint[] }) {
  const values = points
    .map((point) => point.pvk_bo)
    .filter((value): value is number => typeof value === "number");
  const width = 360;
  const height = 120;
  const padding = 12;
  if (values.length === 0) {
    return <p className="text-sm leading-6 text-neutral-500">运行 BO 后会显示 best-so-far 曲线。</p>;
  }

  const min = Math.min(...values) - 0.25;
  const max = Math.max(...values) + 0.25;
  const path = values
    .map((value, index) => {
      const x = padding + (index / Math.max(1, values.length - 1)) * (width - padding * 2);
      const y =
        height -
        padding -
        ((value - min) / Math.max(1, max - min)) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div>
      <svg
        className="w-full rounded-2xl border border-neutral-200 bg-neutral-50"
        viewBox={`0 0 ${width} ${height}`}
      >
        <polyline fill="none" points={path} stroke="#171717" strokeLinecap="round" strokeWidth="4" />
      </svg>
      <p className="mt-2 text-xs leading-5 text-neutral-500">
        曲线来自 session best-so-far；这是 reference lookup，不是湿实验验证。
      </p>
    </div>
  );
}

function StatusPill({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-neutral-200 bg-neutral-50 px-4 py-2 text-xs font-medium text-neutral-600">
      {label}
    </span>
  );
}

function getQuickAction(state: ChatAgentState): { label: string; message: string } | null {
  switch (state) {
    case "awaiting_data_choice":
    case "idle":
      return {
        label: "使用内置 PVK demo 数据",
        message: "使用内置 PVK demo 数据",
      };
    case "awaiting_demo_confirm":
      return {
        label: "确认 demo 数据",
        message: "确认使用内置 PVK demo 数据",
      };
    case "awaiting_goal_confirm":
      return {
        label: "确认目标",
        message: "确认默认 band_alignment / eta 优化目标",
      };
    case "ready_to_run":
      return {
        label: "开始第一轮 BO",
        message: "开始第一轮 BO",
      };
    default:
      return null;
  }
}

function getJsonMap(value: unknown): JsonMap {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonMap) : {};
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function getPrimitiveRecord(value: unknown): Record<string, string | number | boolean | null> {
  const map = getJsonMap(value);
  return Object.fromEntries(
    Object.entries(map).filter((entry): entry is [string, string | number | boolean | null] => {
      const item = entry[1];
      return item === null || ["string", "number", "boolean"].includes(typeof item);
    }),
  );
}

function normalizeToolTrace(trace: OptimizationSession["tool_trace"], currentStep: number): JsonMap[] {
  return (trace || []).map((entry, index) => ({
    name: entry.name || entry.tool || `tool_${index + 1}`,
    arguments: { current_step: currentStep },
    result: entry.message || entry.output,
  }));
}

function normalizeBoCurve(payload: BoCurveResponse | BoCurvePoint[] | undefined): BoCurvePoint[] {
  if (!payload) {
    return [];
  }

  if (Array.isArray(payload)) {
    return payload;
  }

  if (payload.points) {
    return payload.points;
  }

  if (payload.series) {
    if (Array.isArray(payload.series)) {
      return payload.series.map((point) => ({
        iteration: point.iteration,
        pvk_bo: point.best ?? point.objective ?? point.pce,
      }));
    }
    const rows = new Map<number, BoCurvePoint>();
    const points = payload.series.pvk_bo?.points || [];
    for (const point of points) {
      rows.set(point.iteration, {
        ...(rows.get(point.iteration) || { iteration: point.iteration }),
        pvk_bo: point.pce,
      });
    }
    return Array.from(rows.values()).sort((a, b) => a.iteration - b.iteration);
  }

  const iterations = payload.iterations || [];
  return iterations.map((iteration, index) => ({
    iteration,
    pvk_bo: payload.pvk_bo?.[index],
  }));
}

function normalizeChatResponse(response: ChatAgentResponse | ChatResponse): ChatMessage[] {
  if (response.messages && response.messages.length > 0) {
    return response.messages;
  }

  if (response.message) {
    return [response.message];
  }

  if (response.assistant_message) {
    return [{ role: "agent", content: response.assistant_message }];
  }

  if ("reply" in response && response.reply) {
    return [{ role: "agent", content: response.reply }];
  }

  return [{ role: "agent", content: "我已完成工具调用，但没有返回自然语言摘要。" }];
}

function normalizeChatHistory(messages: ChatMessage[]): ChatMessage[] {
  return messages
    .filter((message) => message.content.trim().length > 0)
    .map((message) => ({
      role:
        message.role === "experimenter" || message.role === "user"
          ? "user"
          : message.role === "system"
            ? "system"
            : "assistant",
      content: message.content,
      timestamp: message.timestamp,
    }));
}

function getLatestBest(points: BoCurvePoint[]) {
  const values = points
    .map((point) => point.pvk_bo)
    .filter((value): value is number => typeof value === "number");
  if (values.length === 0) {
    return null;
  }
  return Math.max(...values);
}

function isBoCurveResponse(value: unknown): value is BoCurveResponse {
  return Boolean(value && typeof value === "object");
}

export default App;
