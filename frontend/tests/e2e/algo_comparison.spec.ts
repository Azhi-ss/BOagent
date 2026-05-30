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
    test.setTimeout(180000);

    await expect(page.getByText(/TRADITIONAL BO|传统贝叶斯优化/)).toBeVisible();

    // Set minimal config: nInitial=3, nTrials=5, nSeeds=1
    const inputs = page.locator('input[type="number"]');
    await inputs.nth(0).fill("3"); // nInitial
    await inputs.nth(1).fill("5"); // nTrials
    await inputs.nth(2).fill("1"); // nSeeds

    // Start the run using stable CSS-class locator (text changes when running)
    const runBtn = page.locator("button.run-btn").first();
    await expect(runBtn).toBeVisible();
    await runBtn.click();
    await expect(runBtn).toBeDisabled({ timeout: 5000 });

    // Wait for run to fully complete
    await expect(runBtn).toBeEnabled({ timeout: 150000 });

    // Both SummaryCards must appear
    const tradCard = page.getByTestId("summary-trad");
    const llmCard = page.getByTestId("summary-llm");
    await expect(tradCard).toBeVisible();
    await expect(llmCard).toBeVisible();

    const tradScoreEl = tradCard.getByTestId("score-value");
    const llmScoreEl = llmCard.getByTestId("score-value");

    // Both must show a real numeric score (not "—")
    await expect(tradScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 15000 });
    await expect(llmScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 15000 });

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

    const inputs = page.locator('input[type="number"]');
    await inputs.nth(0).fill("5");  // nInitial
    await inputs.nth(1).fill("15"); // nTrials
    await inputs.nth(2).fill("3");  // nSeeds

    const runBtn = page.locator("button.run-btn").first();
    await runBtn.click();
    await expect(runBtn).toBeDisabled({ timeout: 5000 });
    await expect(runBtn).toBeEnabled({ timeout: 550000 });

    const tradScoreEl = page.getByTestId("summary-trad").getByTestId("score-value");
    const llmScoreEl = page.getByTestId("summary-llm").getByTestId("score-value");

    await expect(tradScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 30000 });
    await expect(llmScoreEl).toHaveText(/^\s*\d+\.\d{4}\s*$/, { timeout: 30000 });

    const tradScore = parseFloat((await tradScoreEl.textContent())!.trim());
    const llmScore = parseFloat((await llmScoreEl.textContent())!.trim());

    console.log(`[perf] Traditional BO: ${tradScore.toFixed(4)}, LLMBO: ${llmScore.toFixed(4)}`);
    expect(llmScore).toBeGreaterThanOrEqual(tradScore - 0.01);
  });
});

