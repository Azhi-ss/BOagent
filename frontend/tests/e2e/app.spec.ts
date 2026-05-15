import { test, expect } from "@playwright/test"
import { AppPage } from "../pages/AppPage"

test.describe("PVK BO Agent Application", () => {
  let appPage: AppPage

  test.beforeEach(async ({ page }) => {
    appPage = new AppPage(page)
    await appPage.goto()
  })

  test("should load application with chat interface", async () => {
    await appPage.assertPageLoaded()
    await appPage.takeScreenshot("app-loaded")
  })

  test("should display quick action button", async () => {
    await expect(appPage.quickActionButton).toBeVisible()
    await appPage.takeScreenshot("quick-action-visible")
  })

  test("should have chat input and send button", async () => {
    await expect(appPage.chatInput).toBeVisible()
    await expect(appPage.sendButton).toBeVisible()
  })
})

test.describe("PVK BO Agent API Tests", () => {
  test.use({
    baseURL: "http://localhost:8000",
  })

  test("should return healthy status from backend API", async ({ request }) => {
    const response = await request.get("/api/v1/health")
    expect(response.ok()).toBeTruthy()

    const data = await response.json()
    expect(data.data).toHaveProperty("status", "ok")
    expect(data.data).toHaveProperty("service", "boagent-api")
  })

  test("should return task list", async ({ request }) => {
    const response = await request.get("/api/v1/tasks")
    expect(response.ok()).toBeTruthy()

    const result = await response.json()
    expect(result).toHaveProperty("data")
    expect(Array.isArray(result.data)).toBeTruthy()
    expect(result.data.length).toBeGreaterThan(0)
  })
})
