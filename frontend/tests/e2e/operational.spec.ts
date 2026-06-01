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
    await expect(page.getByText(/1 · 搜索空间配置|SEARCH SPACE/)).toBeVisible();
    
    // Add a variable
    await page.getByRole("button", { name: /添加变量/ }).click();
    
    // Verify a new variable row appeared
    await expect(page.locator('input[value="Var 3"]')).toBeVisible();
  });
 
  test("should consult agent for suggestions and reasoning", async ({ page }) => {
    // Switch to Operational mode
    test.setTimeout(300000);
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Fill in some history so the agent has data
    await page.getByTestId("op-input-var-Temperature").filter({ visible: true }).fill("120");
    await page.getByTestId("op-input-var-Concentration").filter({ visible: true }).fill("0.2");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("10.1");
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();

    // Click Ask Agent
    const askBtn = page.getByTestId("op-suggest-btn").filter({ visible: true });
    await askBtn.click();
    
    // Wait for the response (this might take a while due to LLM call)
    // Reasoning can take 30-60s or more depending on model load
    await expect(page.getByText(/AGENT 优化推理决策|REASONING/)).toBeVisible({ timeout: 150000 });
    await expect(page.getByText(/建议配方|SUGGESTIONS/).filter({ visible: true }).first()).toBeVisible({ timeout: 30000 });
    
    // Check if at least one suggestion is shown
    const firstSuggestion = page.getByText(/★ 首选建议配方|TOP RECOMMENDATION/);
    await expect(firstSuggestion).toBeVisible();

    // Click the first suggestion card
    await firstSuggestion.click();
    
    // Check if the input fields were updated
    const tempInput = page.getByTestId("op-input-var-Temperature");
    const tempVal = await tempInput.getAttribute("value");
    expect(Number(tempVal)).toBeGreaterThan(0);
  });
 
  test("should allow adding manual observations", async ({ page }) => {
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Fill in manual observation using stable data-testid
    await page.getByTestId("op-input-var-Temperature").filter({ visible: true }).fill("150");
    await page.getByTestId("op-input-var-Concentration").filter({ visible: true }).fill("0.5");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("15.5");
    
    // Add
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();
    
    // Verify it appeared in history
    await expect(page.getByText("15.5000")).toBeVisible();
    await expect(page.getByText("150.000")).toBeVisible();
    await expect(page.getByText("0.500")).toBeVisible();
  });

  test("should render the Landscape Canvas with data", async ({ page }) => {
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Check if Canvas title is visible
    await expect(page.getByText(/优化地形投影|OPTIMIZATION LANDSCAPE/)).toBeVisible();
    
    // Add a manual observation to populate the canvas
    await page.getByTestId("op-input-var-Temperature").filter({ visible: true }).fill("120");
    await page.getByTestId("op-input-var-Concentration").filter({ visible: true }).fill("0.8");
    await page.getByTestId("op-input-score").filter({ visible: true }).fill("18.2");
    await page.getByTestId("op-add-btn").filter({ visible: true }).click();
    
    // Verify the canvas contains svg elements (Scatter points)
    const canvas = page.locator('section:has-text("优化地形投影")');
    await expect(canvas.locator('svg')).toBeVisible();
    
    // Check for legend items
    await expect(canvas.getByText("已观测")).toBeVisible();
    await expect(canvas.getByText("当前最佳")).toBeVisible();
  });
});
