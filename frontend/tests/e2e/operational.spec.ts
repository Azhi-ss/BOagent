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
 
  test("should perform an agent consultation", async ({ page }) => {
    // Switch to Operational mode
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Click Ask Agent
    const askBtn = page.getByRole("button", { name: /咨询 Agent/i });
    await askBtn.click();
    
    // Wait for the response (this might take a while due to LLM call)
    await expect(page.getByText(/AGENT 优化推理决策|REASONING/)).toBeVisible({ timeout: 60000 });
    await expect(page.getByText(/推荐推荐配方|SUGGESTIONS/)).toBeVisible();
    
    // Check if at least one suggestion is shown
    await expect(page.getByText(/★ 首选建议配方|TOP RECOMMENDATION/)).toBeVisible();
  });
 
  test("should allow adding manual observations", async ({ page }) => {
    await page.getByRole("button", { name: /实验实操|OPERATIONAL/ }).click();
    
    // Use visible locators to avoid picking up inputs from hidden BenchMode
    const visibleInputs = page.locator('input[type="number"]:visible');
    
    // Fill in manual observation
    // The first 4 visible number inputs are Min/Max for the 2 search space variables
    // Manual entry starts at index 4 of visible inputs
    await visibleInputs.nth(4).fill("150"); 
    await visibleInputs.nth(5).fill("0.5");
    await visibleInputs.nth(6).fill("15.5");
    
    // Add
    await page.getByRole("button", { name: /录入/ }).click();
    
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
    const visibleInputs = page.locator('input[type="number"]:visible');
    await visibleInputs.nth(4).fill("120"); // Var 1
    await visibleInputs.nth(5).fill("0.8"); // Var 2
    await visibleInputs.nth(6).fill("18.2"); // Score
    await page.getByRole("button", { name: /录入/ }).click();
    
    // Verify the canvas contains svg elements (Scatter points)
    const canvas = page.locator('section:has-text("优化地形投影")');
    await expect(canvas.locator('svg')).toBeVisible();
    
    // Check for legend items
    await expect(canvas.getByText("已观测")).toBeVisible();
    await expect(canvas.getByText("当前最佳")).toBeVisible();
  });
});
