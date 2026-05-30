import { test, expect } from "@playwright/test";

test.describe("Benchmark Mode E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("BO·AGENT")).toBeVisible();
  });

  test("should perform a benchmark run", async ({ page }) => {
    // LLM API calls take ~6s each; nTrials=3 nSeeds=2 = minimum viable run.
    // First aggregate arrives after the first iteration (~6-15s per engine call).
    test.setTimeout(180000);

    // We are in Bench mode by default
    await expect(page.getByText(/TRADITIONAL BO|传统贝叶斯优化/)).toBeVisible();

    // Use Fast Preview preset values: nTrials=3, nSeeds=2 for minimal LLM cost.
    const visibleInputs = page.locator('input[type="number"]');
    await visibleInputs.nth(1).fill("3");  // nTrials
    await visibleInputs.nth(2).fill("2");  // nSeeds

    // Start the comparison
    const runBtn = page.getByRole("button", { name: /开启对比实验/ });
    await runBtn.click();

    // Confirm the running indicator appears immediately
    await expect(page.getByText("运行中...")).toBeVisible();

    // After the first iteration completes the backend pushes an aggregate event via
    // _iter_snapshot → the convergence chart should become visible.
    // This validates the per-iteration streaming fix (not just end-of-seed).
    await expect(page.locator(".recharts-wrapper, svg.recharts-surface").first())
      .toBeVisible({ timeout: 120000 });

    // Stop the run
    await page.getByRole("button", { name: "停止" }).click();
    await expect(page.getByRole("button", { name: /开启对比实验/ })).toBeEnabled();
  });
});
