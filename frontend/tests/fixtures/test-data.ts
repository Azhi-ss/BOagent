export const TEST_MESSAGES = {
  greetings: ["你好", "Hello", "hi"],
  capabilities: ["你能做什么", "介绍一下系统", "What can you do?"],
  optimizationRequests: [
    "优化钙钛矿钝化配方",
    "推荐3个能带对齐的候选",
    "Optimize perovskite passivation",
  ],
  questions: ["这个结果的置信度如何？", "有什么风险？", "下一轮是什么？"],
} as const

export const TEST_TASKS = {
  passivationDemo: "passivation_demo",
  bandAlignment: "band_alignment",
  defectsDoping: "defects_doping",
} as const

export const E2E_TIMEOUTS = {
  SHORT: 5000,
  MEDIUM: 15000,
  LONG: 30000,
} as const

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
