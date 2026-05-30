import { Page, Locator, expect } from "@playwright/test"

export class AppPage {
  readonly page: Page
  readonly title: Locator
  readonly taskSelector: Locator
  readonly runButton: Locator
  readonly boCurveContainer: Locator
  readonly modeEvaluationButton: Locator
  readonly modeOperationalButton: Locator

  constructor(page: Page) {
    this.page = page
    this.title = page.getByRole("heading", { name: "BO·AGENT" })
    this.taskSelector = page.locator("select").first()
    this.runButton = page.getByRole("button", { name: /开启对比实验/ })
    this.boCurveContainer = page.locator("svg")
    this.modeEvaluationButton = page.getByRole("button", { name: /性能评测/ })
    this.modeOperationalButton = page.getByRole("button", { name: /实验实操/ })
  }

  async goto() {
    await this.page.goto("/")
    await this.page.waitForLoadState("networkidle")
  }

  async assertPageLoaded() {
    await expect(this.title).toBeVisible()
    await expect(this.runButton).toBeVisible()
  }

  async selectTask(taskId: string) {
    await this.taskSelector.selectOption(taskId)
  }

  async assertBoCurveVisible() {
    await expect(this.boCurveContainer).toBeVisible()
  }

  async takeScreenshot(name: string) {
    await this.page.screenshot({
      path: `playwright-report/artifacts/${name}.png`,
      fullPage: true,
    })
  }
}

