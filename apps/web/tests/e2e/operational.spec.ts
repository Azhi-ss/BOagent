import { test, expect } from "@playwright/test";

test.describe("Operational Mode E2E", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app
    await page.goto("/");
    // Wait for the app to load
    await expect(page.getByText("BO·AGENT")).toBeVisible();
  });

  test("should switch to operational mode and add a variable", async ({ page }) => {
    // Switch to Operational mode
    const operationalBtn = page.getByRole("button", { name: /实验实操|OPERATIONAL/ });
    await operationalBtn.click();
    
    // Verify Operational Mode is active
    await expect(page.getByText(/1 · 实验空间设计|SEARCH SPACE/)).toBeVisible();
    
    // Add a variable
    const addBtn = page.getByRole("button", { name: /添加变量/ });
    await expect(addBtn).toBeVisible();
    await addBtn.click();
    
    // Verify a new variable row appeared (originally Var 3, now Var 5 as we have 4 default variables)
    await expect(page.locator('input[value="Var 5"]')).toBeVisible();
  });
 
  test("should consult agent for suggestions and reasoning", async ({ page }) => {
    // Intercept and mock the API request to operational/suggest for reliable offline test runs
    await page.route(/.*\/api\/v1\/operational\/suggest.*/, async (route) => {
      if (route.request().method() === "OPTIONS") {
        await route.fulfill({
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
          }
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          headers: {
            "Access-Control-Allow-Origin": "*",
          },
          body: JSON.stringify({
            data: {
              suggestions: [
                {
                  "x_3MTPAI": 0.45,
                  "x_PDAI2": 0.15,
                  "x_EDAI2": 0.35,
                  "x_PipDI": 0.05,
                  "CBO": 0.1,
                  "CBO_Status": "Ideal",
                  "VBO": 1.7,
                  "VBO_Status": "Ideal"
                },
                {
                  "x_3MTPAI": 0.30,
                  "x_PDAI2": 0.20,
                  "x_EDAI2": 0.40,
                  "x_PipDI": 0.10,
                  "CBO": -0.3,
                  "CBO_Status": "Cliff (Recombination Loss)",
                  "VBO": 1.2,
                  "VBO_Status": "Sub-optimal"
                }
              ],
              analysis: "AGENT 优化推理决策 (REASONING)\n\n优先推荐首选配方 1。其配方完全符合组分比例守恒，能有效平衡载流子提取与抑制缺陷复合。",
              prompt: "Mock prompt"
            }
          })
        });
      }
    });

    // Switch to Operational mode
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Fill in manual entry to test suggest click
    await page.getByTestId("op-input-var-x_3MTPAI").filter({ visible: true }).fill("0.45");
    await page.getByTestId("op-input-var-x_PDAI2").filter({ visible: true }).fill("0.15");
    await page.getByTestId("op-input-var-x_EDAI2").filter({ visible: true }).fill("0.35");
    await page.getByTestId("op-input-var-x_PipDI").filter({ visible: true }).fill("0.05");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("26.0");
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();

    // Click Ask Agent
    const askBtn = page.getByTestId("op-suggest-btn").filter({ visible: true });
    await askBtn.click();
    
    // Wait for the response (this will be instant now due to the Playwright route mock)
    await expect(page.getByRole("heading", { name: /AGENT 优化推理决策|REASONING/ })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText(/建议配方|SUGGESTIONS/).filter({ visible: true }).first()).toBeVisible({ timeout: 15000 });
    
    // Check if at least one suggestion is shown
    const firstSuggestion = page.getByText(/★ 首选建议配方|TOP RECOMMENDATION/);
    await expect(firstSuggestion).toBeVisible();

    // Click the first suggestion card
    await firstSuggestion.click();
    
    // Check if the input fields were updated
    const tempInput = page.getByTestId("op-input-var-x_3MTPAI");
    const tempVal = await tempInput.getAttribute("value");
    expect(Number(tempVal)).toBeGreaterThan(0);
  });
 
  test("should allow adding manual observations", async ({ page }) => {
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Fill in manual observation using stable data-testid
    const chPvk = page.getByTestId("op-input-var-x_3MTPAI").filter({ visible: true });
    await expect(chPvk).toBeVisible();
    await chPvk.fill("0.50");
    await page.getByTestId("op-input-var-x_PDAI2").filter({ visible: true }).fill("0.10");
    await page.getByTestId("op-input-var-x_EDAI2").filter({ visible: true }).fill("0.30");
    await page.getByTestId("op-input-var-x_PipDI").filter({ visible: true }).fill("0.10");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("26.2");
    
    // Add
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();
    
    // Verify it appeared in history
    await expect(page.getByText("26.2000")).toBeVisible();
    await expect(page.getByText("0.500")).toBeVisible();
    await expect(page.getByText("0.100").first()).toBeVisible();
  });

  test("should render the Landscape Canvas with data", async ({ page }) => {
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Check if Canvas title is visible
    await expect(page.getByText(/优化地形投影|OPTIMIZATION LANDSCAPE/)).toBeVisible();
    
    // Add a manual observation to populate the canvas
    const chPvk = page.getByTestId("op-input-var-x_3MTPAI").filter({ visible: true });
    await expect(chPvk).toBeVisible();
    await chPvk.fill("0.45");
    await page.getByTestId("op-input-var-x_PDAI2").filter({ visible: true }).fill("0.15");
    await page.getByTestId("op-input-var-x_EDAI2").filter({ visible: true }).fill("0.35");
    await page.getByTestId("op-input-var-x_PipDI").filter({ visible: true }).fill("0.05");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("26.0");
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();
    
    // Verify the canvas contains svg elements (Scatter points)
    const canvas = page.locator('section:has-text("优化地形投影")');
    await expect(canvas.locator('svg')).toBeVisible();
    
    // Check for legend items
    await expect(canvas.getByText("已观测")).toBeVisible();
    await expect(canvas.getByText("当前最佳")).toBeVisible();
  });
});
