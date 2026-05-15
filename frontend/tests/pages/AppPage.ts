import { Page, Locator, expect } from "@playwright/test"

export class AppPage {
  readonly page: Page
  readonly welcomeMessage: Locator
  readonly chatInput: Locator
  readonly sendButton: Locator
  readonly taskSelector: Locator
  readonly dataBoundaryBanner: Locator
  readonly boCurveContainer: Locator
  readonly candidatePanel: Locator
  readonly toolTracePanel: Locator
  readonly metricCards: Locator
  readonly chatBubbles: Locator
  readonly quickActionButton: Locator

  constructor(page: Page) {
    this.page = page
    this.welcomeMessage = page.getByText(/PVK BO 研究助理/)
    this.chatInput = page.locator('textarea[placeholder*="用内置 reference 数据跑一轮"]')
    this.sendButton = page.locator('button[type="submit"]')
    this.taskSelector = page.locator("select").first()
    this.dataBoundaryBanner = page.getByText(/reference data 不是湿实验/)
    this.boCurveContainer = page.locator("svg")
    this.candidatePanel = page.getByText(/候选配方|Candidate/)
    this.toolTracePanel = page.getByText(/工具调用/)
    this.metricCards = page.getByText(/迭代|Iteration/)
    this.chatBubbles = page.locator("article")
    this.quickActionButton = page.getByRole("button", { name: /使用内置 PVK demo 数据/ })
  }

  async goto() {
    await this.page.goto("/")
    await this.page.waitForLoadState("networkidle")
  }

  async sendMessage(message: string) {
    await this.chatInput.fill(message)
    await this.sendButton.click()
  }

  async selectTask(taskId: string) {
    await this.taskSelector.selectOption(taskId)
  }

  async waitForMessageContaining(text: string, timeout: number = 30000) {
    await this.page.waitForFunction(
      (expectedText) => document.body.textContent?.includes(expectedText),
      text,
      { timeout },
    )
  }

  async getMessageCount() {
    return await this.chatMessages.count()
  }

  async assertPageLoaded() {
    await expect(this.chatInput).toBeVisible()
    await expect(this.sendButton).toBeVisible()
  }

  async assertDataBoundaryVisible() {
    await expect(this.dataBoundaryBanner).toBeVisible()
  }

  async assertBoCurveVisible() {
    await expect(this.boCurveContainer).toBeVisible()
  }

  async getBubbleCount() {
    return await this.chatBubbles.count()
  }

  async takeScreenshot(name: string) {
    await this.page.screenshot({
      path: `playwright-report/artifacts/${name}.png`,
      fullPage: true,
    })
  }
}
