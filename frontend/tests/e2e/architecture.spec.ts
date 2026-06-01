import { test, expect } from "@playwright/test"

test.describe("Architecture Documentation E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Log page errors and messages for debugging
    page.on("pageerror", (err) => {
      console.error("[Browser Error]", err.message)
    })
    page.on("console", (msg) => {
      console.log("[Browser Console]", msg.text())
    })

    // Navigate to /architecture served by FastAPI backend
    await page.goto("http://localhost:8000/architecture")
  })

  test("should load the architecture page with proper title and elements", async ({ page }) => {
    // Verify title/header is present
    await expect(page.locator("h1")).toContainText("LLMBO 混合系统架构")

    // Default view state: Pipeline Flowchart should be visible, Layers view should be hidden
    await expect(page.locator("#view-pipeline")).toBeVisible()
    await expect(page.locator("#view-layers")).toBeHidden()
  })

  test("should update details sidebar when clicking pipeline nodes", async ({ page }) => {
    // Select a node (e.g. backend) in the pipeline flowchart view
    // Use force: true to prevent pointer-event interception by overlapping SVG text elements
    const backendRect = page.locator("#node-backend rect").first()
    await expect(backendRect).toBeVisible()
    await backendRect.click({ force: true })

    // Assert details panel updates appropriately with the correct Chinese title
    const detailsTitle = page.locator("#details-title")
    await expect(detailsTitle).toContainText("FastAPI 后端 Orchestrator")

    const codeFile = page.locator("#code-file")
    await expect(codeFile).toContainText("backend/api.py")
  })

  test("should switch views and interact with system layers", async ({ page }) => {
    // Click button to switch to Layers View
    const layersBtn = page.locator("#btn-layers")
    await expect(layersBtn).toBeVisible()
    await layersBtn.click({ force: true })

    // Assert visibility switches
    await expect(page.locator("#view-layers")).toBeVisible()
    await expect(page.locator("#view-pipeline")).toBeHidden()

    // Click a layer (e.g. layer-memory) in the Layers View
    const memoryPolygon = page.locator("#layer-memory polygon")
    await expect(memoryPolygon).toBeVisible()
    await memoryPolygon.click({ force: true })

    // Assert details panel updates for the clicked layer
    const detailsTitle = page.locator("#details-title")
    await expect(detailsTitle).toContainText("向量记忆 (VectorMemory RAG)")
  })

  test("should run the interactive calculator simulator", async ({ page }) => {
    // Check initial score value
    const scoreVal = page.locator("#res-score")
    await expect(scoreVal).toBeVisible()
    const initialScoreText = await scoreVal.textContent()
    expect(initialScoreText).not.toBeNull()

    // Adjust a slider, e.g. slider-std, via page.evaluate since it is a range input
    await page.evaluate(() => {
      const slider = document.getElementById("slider-std") as HTMLInputElement
      if (slider) {
        slider.value = "3.0"
        slider.dispatchEvent(new Event("input"))
      }
    })

    // Confirm that the calculated value changes
    await expect(scoreVal).not.toHaveText(initialScoreText || "")
  })
})
