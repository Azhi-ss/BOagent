import { test, expect } from "@playwright/test"
import { AppPage } from "../pages/AppPage"

test.describe("Chat Flow - Critical User Journey", () => {
  let appPage: AppPage

  test.beforeEach(async ({ page }) => {
    appPage = new AppPage(page)
    await appPage.goto()
  })

  test("should send message via quick action button", async ({ page }) => {
    // Step 1: Verify initial state
    await appPage.assertPageLoaded()
    await appPage.takeScreenshot("step1-initial-state")

    const initialCount = await appPage.getBubbleCount()

    // Step 2: Click quick action button
    await appPage.quickActionButton.click()
    await page.waitForTimeout(2000)

    // Step 3: Verify a new message appeared
    const finalCount = await appPage.getBubbleCount()
    expect(finalCount).toBeGreaterThan(initialCount)

    await appPage.takeScreenshot("step2-after-quick-action")
  })

  test("should send user message and receive response", async ({ page }) => {
    const initialCount = await appPage.getBubbleCount()

    // Send a message
    await appPage.sendMessage("你好，请介绍一下这个系统")

    // Wait for response
    await page.waitForTimeout(3000)

    const finalCount = await appPage.getBubbleCount()
    expect(finalCount).toBeGreaterThan(initialCount)

    await appPage.takeScreenshot("chat-message-response")
  })

  test("should display chat bubbles correctly", async ({ page }) => {
    await appPage.sendMessage("测试消息")
    await page.waitForTimeout(2000)

    // Verify chat bubbles exist
    const bubbles = await appPage.getBubbleCount()
    expect(bubbles).toBeGreaterThan(0)

    await appPage.takeScreenshot("chat-bubbles")
  })
})

test.describe("Edge Cases", () => {
  let appPage: AppPage

  test.beforeEach(async ({ page }) => {
    appPage = new AppPage(page)
    await appPage.goto()
  })

  test("should handle very long message", async ({ page }) => {
    test.fixme() // Disabled - requires proper error handling
    const longMessage = "测试".repeat(50)
    await appPage.sendMessage(longMessage)
    await page.waitForTimeout(2000)
    await appPage.takeScreenshot("long-message")
  })
})
