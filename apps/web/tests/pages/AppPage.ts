import { Page, Locator, expect } from "@playwright/test"

export class AppPage {
  readonly page: Page
  readonly title: Locator
  readonly taskSelector: Locator
  readonly runButton: Locator
  readonly stopButton: Locator
  readonly boCurveContainer: Locator
  readonly modeEvaluationButton: Locator
  readonly modeOperationalButton: Locator

  constructor(page: Page) {
    this.page = page
    this.title = page.getByRole("heading", { name: "BO·AGENT" })
    this.taskSelector = page.getByTestId("task-selector").filter({ visible: true })
    this.runButton = page.getByTestId("run-bench-btn").filter({ visible: true })
    this.stopButton = page.getByTestId("stop-bench-btn").filter({ visible: true })
    this.boCurveContainer = page.locator("svg").filter({ visible: true })
    this.modeEvaluationButton = page.getByRole("button", { name: /性能评测/ }).filter({ visible: true })
    this.modeOperationalButton = page.getByRole("button", { name: /实验实操/ }).filter({ visible: true })
  }

  async goto() {
    await this.page.goto("/")
    await this.page.waitForLoadState("networkidle")
  }

  async assertPageLoaded() {
    await expect(this.title).toBeVisible()
    await expect(this.runButton).toBeVisible()
  }

  async runBenchmark() {
    await this.runButton.click()
  }

  async stopBenchmark() {
    await this.stopButton.click()
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

