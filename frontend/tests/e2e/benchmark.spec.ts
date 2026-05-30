import { test, expect } from "@playwright/test";

test.describe("Benchmark Mode E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("BO·AGENT")).toBeVisible();
  });

  test("should perform a benchmark run", async ({ page }) => {
    // We are in Bench mode by default
    await expect(page.getByText("A · TRADITIONAL BO")).toBeVisible();
    
    // Click Run Comparison
    const runBtn = page.getByRole("button", { name: "▶ RUN COMPARISON" });
    await runBtn.click();
    
    // Check if it starts running
    await expect(page.getByText("RUNNING…")).toBeVisible();
    
    // Wait for at least one iteration to complete
    // The chart or the metrics should update
    await expect(page.getByText("ITER 1")).toBeVisible({ timeout: 120000 });
    
    // Stop the run
    await page.getByRole("button", { name: "STOP" }).click();
    await expect(page.getByRole("button", { name: "▶ RUN COMPARISON" })).toBeEnabled();
  });
});
