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
    const operationalBtn = page.getByRole("button", { name: "OPERATIONAL" });
    await operationalBtn.click();
    
    // Verify Operational Mode is active
    await expect(page.getByText("1 · SEARCH SPACE")).toBeVisible();
    
    // Add a variable
    await page.getByRole("button", { name: "+ ADD VARIABLE" }).click();
    
    // Verify a new variable row appeared
    await expect(page.locator('input[value="Var 3"]')).toBeVisible();
  });

  test("should perform an agent consultation", async ({ page }) => {
    // Switch to Operational mode
    await page.getByRole("button", { name: "OPERATIONAL" }).click();
    
    // Click Ask Agent
    const askBtn = page.getByRole("button", { name: "ASK AGENT FOR NEXT EXPERIMENT" });
    await askBtn.click();
    
    // Wait for the response (this might take a while due to LLM call)
    // We expect "AGENT REASONING" to appear
    await expect(page.getByText("AGENT REASONING")).toBeVisible({ timeout: 60000 });
    await expect(page.getByText("SUGGESTED FORMULATIONS")).toBeVisible();
    
    // Check if at least one suggestion is shown
    await expect(page.getByText("★ TOP RECOMMENDATION")).toBeVisible();
  });

  test("should allow adding manual observations", async ({ page }) => {
    await page.getByRole("button", { name: "OPERATIONAL" }).click();
    
    // Use visible locators to avoid picking up inputs from hidden BenchMode
    const visibleInputs = page.locator('input[type="number"]:visible');
    
    // Fill in manual observation
    // The first 4 visible number inputs are Min/Max for the 2 search space variables
    // Manual entry starts at index 4 of visible inputs
    await visibleInputs.nth(4).fill("150"); 
    await visibleInputs.nth(5).fill("0.5");
    await visibleInputs.nth(6).fill("15.5");
    
    // Add
    await page.getByRole("button", { name: "ADD", exact: true }).click();
    
    // Verify it appeared in history
    await expect(page.getByText("15.5000")).toBeVisible();
    await expect(page.getByText("150.000")).toBeVisible();
    await expect(page.getByText("0.500")).toBeVisible();
  });
});
