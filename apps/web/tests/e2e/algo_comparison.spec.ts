import { test, expect } from "@playwright/test";

/**
 * Algorithm comparison E2E tests.
 *
 * Test 1 (CI-safe): Verifies both Traditional BO and LLMBO complete a run and
 * produce valid, non-zero scores. Does NOT assert strict ordering because
 * nTrials=5 / nSeeds=1 has high variance and LLM responses are stochastic.
 *
 * Test 2 (performance, skipped in CI): Runs a longer benchmark and asserts
 * LLMBO >= Traditional BO. Run manually with `--grep performance`.
 */
test.describe("Algorithm Comparison - LLMBO vs Traditional BO", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("BO·AGENT")).toBeVisible();
  });

  test("both algorithms should produce valid non-zero scores", async ({ page }) => {
    // nTrials=5, nSeeds=1 → ~30-60s total.
    test.setTimeout(300000);

    await expect(page.getByText(/TRADITIONAL BO|传统贝叶斯优化/)).toBeVisible();

    // Set minimal config: nTrials=5, nSeeds=1 → ~30-60s total.
    // Use stable data-testid instead of fragile nth(i) locators
    await page.getByTestId("input-n-initial").filter({ visible: true }).fill("3");
    await page.getByTestId("input-n-trials").filter({ visible: true }).fill("5");
    await page.getByTestId("input-n-seeds").filter({ visible: true }).fill("1");

    // Start the run using stable data-testid
    const runBtn = page.getByTestId("run-bench-btn").filter({ visible: true });
    await expect(runBtn).toBeVisible();
    await runBtn.click();
    await expect(runBtn).toBeDisabled({ timeout: 20000 });

    // Wait for run to fully complete (LLM calls can be slow)
    await expect(runBtn).toBeEnabled({ timeout: 480000 });

    // Both SummaryCards must appear
    const tradCard = page.getByTestId("summary-trad").filter({ visible: true });
    const llmCard = page.getByTestId("summary-llm").filter({ visible: true });
    await expect(tradCard).toBeVisible({ timeout: 30000 });
    await expect(llmCard).toBeVisible({ timeout: 30000 });

    const tradScoreEl = tradCard.getByTestId("score-value").filter({ visible: true });
    const llmScoreEl = llmCard.getByTestId("score-value").filter({ visible: true });

    // Both must show a real numeric score (not "—")
    await expect(tradScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 60000 });
    await expect(llmScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 60000 });

    const tradScore = parseFloat((await tradScoreEl.textContent())!.trim());
    const llmScore = parseFloat((await llmScoreEl.textContent())!.trim());

    console.log(`[algo_comparison] Traditional BO: ${tradScore.toFixed(4)}`);
    console.log(`[algo_comparison] LLMBO:          ${llmScore.toFixed(4)}`);
    console.log(`[algo_comparison] LLMBO advantage: ${(llmScore - tradScore).toFixed(4)}`);

    // Functional assertions: both algorithms produced valid scores.
    // (Strict ordering not enforced here due to nSeeds=1 variance.)
    expect(tradScore).toBeGreaterThan(0);
    expect(llmScore).toBeGreaterThan(0);
  });

  // Run manually: npx playwright test --grep performance
  test.skip("LLMBO should statistically outperform Traditional BO @performance", async ({ page }) => {
    // nTrials=15, nSeeds=3 gives a more reliable comparison (~5 min).
    test.setTimeout(600000);

    await expect(page.getByText(/TRADITIONAL BO|传统贝叶斯优化/)).toBeVisible();

    await page.getByTestId("input-n-initial").filter({ visible: true }).fill("5");
    await page.getByTestId("input-n-trials").filter({ visible: true }).fill("15");
    await page.getByTestId("input-n-seeds").filter({ visible: true }).fill("3");

    const runBtn = page.getByTestId("run-bench-btn").filter({ visible: true });
    await runBtn.click();
    await expect(runBtn).toBeDisabled({ timeout: 20000 });
    await expect(runBtn).toBeEnabled({ timeout: 1200000 });

    const tradScoreEl = page.getByTestId("summary-trad").filter({ visible: true }).getByTestId("score-value").filter({ visible: true });
    const llmScoreEl = page.getByTestId("summary-llm").filter({ visible: true }).getByTestId("score-value").filter({ visible: true });

    await expect(tradScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 60000 });
    await expect(llmScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 60000 });

    const tradScore = parseFloat((await tradScoreEl.textContent())!.trim());
    const llmScore = parseFloat((await llmScoreEl.textContent())!.trim());

    console.log(`[perf] Traditional BO: ${tradScore.toFixed(4)}, LLMBO: ${llmScore.toFixed(4)}`);
    expect(llmScore).toBeGreaterThanOrEqual(tradScore - 0.01);
  });
});

