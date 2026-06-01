import { test, expect } from "@playwright/test";

test.describe("Benchmark Mode E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("BO·AGENT")).toBeVisible();
  });

  test("should perform a benchmark run", async ({ page }) => {
    // LLM API calls take ~6s each; nTrials=3 nSeeds=2 = minimum viable run.
    // First aggregate arrives after the first iteration (~6-15s per engine call).
    test.setTimeout(400000);

    // We are in Bench mode by default
    await expect(page.getByText(/TRADITIONAL BO|传统贝叶斯优化/)).toBeVisible();

    // Use Fast Preview preset values: nTrials=3, nSeeds=2 for minimal LLM cost.
    // Use stable data-testid with visibility filter
    await page.getByTestId("input-n-trials").filter({ visible: true }).fill("3");
    await page.getByTestId("input-n-seeds").filter({ visible: true }).fill("2");

    // Start the comparison
    const runBtn = page.getByTestId("run-bench-btn").filter({ visible: true });
    await runBtn.click();

    // Confirm the running indicator appears immediately
    await expect(page.getByText("运行中...").filter({ visible: true })).toBeVisible({ timeout: 20000 });

    // After the first iteration completes the backend pushes an aggregate event via
    // _iter_snapshot → the convergence chart should become visible.
    // This validates the per-iteration streaming fix (not just end-of-seed).
    // LLM calls can be slow, especially in parallel seeds.
    await expect(page.locator(".recharts-wrapper, svg.recharts-surface").filter({ visible: true }).first())
      .toBeVisible({ timeout: 300000 });

    // Stop the run
    const stopBtn = page.getByTestId("stop-bench-btn").filter({ visible: true });
    await stopBtn.click();
    await expect(runBtn).toBeEnabled({ timeout: 30000 });
  });
});
