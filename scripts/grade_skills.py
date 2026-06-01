#!/usr/bin/env python3
"""LLM-as-a-Judge evaluation with dual-judge consensus (auto + pro).

For each test case in evals/evals.json:
  1. Generate With-Skill response via smart_gemini.sh
  2. Judge A (auto/Flash): evaluate assertions
  3. Judge B (pro): evaluate assertions
  4. Consensus: 2/2 agree = final, 1/1 split = pro wins
  5. Baseline: single auto judge (saves quota)
"""
import os
import sys
import json
import re
import subprocess

ANTI_HANG_GUARD = (
    "\n\nIMPORTANT: Do NOT write, edit, delete, or create any files in the workspace. "
    "Do NOT call any modifying tools (such as write_file, replace_file_content, etc.). "
    "Do NOT run any long-running or background processes (like uvicorn, python -m uvicorn, "
    "npm run dev, vite, or any development servers). "
    "Just analyze the code and write your explanation directly as text. "
    "Do NOT ask the user for confirmation or input. Just write the answer as text and exit."
)


def read_skill_content(skill_name):
    skill_path = f".agents/skills/{skill_name}/SKILL.md"
    if not os.path.exists(skill_path):
        return ""
    with open(skill_path, "r", encoding="utf-8") as f:
        return f.read()


def run_gemini_cli(prompt, model=None):
    """Run gemini via smart_gemini.sh (auto) or direct gemini -m pro."""
    # Append guard unless it's a judge prompt (which already has strict instructions)
    if "You are a professional AI judge" not in prompt:
        prompt = prompt + ANTI_HANG_GUARD

    if model == "pro":
        cmd = ["gemini", "-m", "pro", "-p", prompt, "--approval-mode", "plan", "--skip-trust"]
    else:
        script_path = "./scripts/smart_gemini.sh"
        if not os.path.exists(script_path):
            script_path = "scripts/smart_gemini.sh"
        cmd = [script_path, prompt]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=240)
        return res.stdout
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Command timed out (model={model or 'auto'})", file=sys.stderr)
        return ""
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ Error (model={model or 'auto'}): {e.stderr[:200]}", file=sys.stderr)
        return e.stdout if e.stdout else ""


def build_judge_prompt(prompt, response, assertions):
    """Build the judge evaluation prompt."""
    assertions_str = ""
    for idx, ast in enumerate(assertions):
        assertions_str += f"{idx+1}. Name: {ast['name']}\n   Description: {ast['description']}\n\n"

    return f"""You are a professional AI judge. Your task is to evaluate an AI assistant's response against a list of assertions.

User Prompt:
{prompt}

Model Response to evaluate:
\"\"\"
{response}
\"\"\"

Assertions to check:
{assertions_str}

For each assertion, read the model response carefully. Determine whether the response satisfies the description and logic.
You MUST output a valid JSON block containing your evaluation.
Format:
```json
{{
  "assertions": [
    {{
      "name": "assertion_name_1",
      "passed": true,
      "reason": "Explain why it passed"
    }},
    {{
      "name": "assertion_name_2",
      "passed": false,
      "reason": "Explain why it failed"
    }}
  ]
}}
```
Keep your output strictly as a JSON block inside the markdown fence. Do not output other conversational text."""


def parse_judge_output(judge_output, assertions):
    """Parse judge JSON output. Returns dict with 'assertions' list."""
    match = re.search(r'\{.*\}', judge_output, re.DOTALL)
    if not match:
        print(f"  ⚠️ Failed to parse JSON from judge output", file=sys.stderr)
        return {
            "assertions": [{"name": a["name"], "passed": False, "reason": "Failed to parse judge JSON"} for a in assertions]
        }
    try:
        return json.loads(match.group(0))
    except Exception as e:
        print(f"  ⚠️ JSON decode error: {e}", file=sys.stderr)
        return {
            "assertions": [{"name": a["name"], "passed": False, "reason": "JSON decode error"} for a in assertions]
        }


def consensus_merge(judge_a_result, judge_b_result, assertions):
    """Merge two judge results using consensus rules.

    2/2 agree → final result
    1/1 split → pro (judge B) wins
    """
    a_lookup = {a["name"]: a for a in judge_a_result.get("assertions", [])}
    b_lookup = {a["name"]: a for a in judge_b_result.get("assertions", [])}

    merged = []
    for ast in assertions:
        name = ast["name"]
        a_result = a_lookup.get(name, {"passed": False, "reason": "No evaluation"})
        b_result = b_lookup.get(name, {"passed": False, "reason": "No evaluation"})

        a_passed = a_result.get("passed", False)
        b_passed = b_result.get("passed", False)

        if a_passed == b_passed:
            # Consensus: both agree
            merged.append({
                "name": name,
                "passed": a_passed,
                "reason": f"[Consensus 2/2] {a_result.get('reason', '')}",
                "judge_a": a_passed,
                "judge_b": b_passed,
            })
        else:
            # Split: pro (judge B) wins
            merged.append({
                "name": name,
                "passed": b_passed,
                "reason": f"[Pro Override 1/1] Auto={'Pass' if a_passed else 'Fail'}, Pro={'Pass' if b_passed else 'Fail'}. Pro reason: {b_result.get('reason', '')}",
                "judge_a": a_passed,
                "judge_b": b_passed,
            })

    return merged


def main():
    evals_path = "evals/evals.json"
    if not os.path.exists(evals_path):
        print(f"Evals file not found at {evals_path}")
        sys.exit(1)

    with open(evals_path, "r", encoding="utf-8") as f:
        evals_data = json.load(f)

    eval_runs = evals_data.get("evals", [])
    results = []

    print(f"Loaded {len(eval_runs)} test cases from {evals_path}")
    print("Running Dual-Judge Consensus Evaluation (auto + pro)")
    print("=" * 60)

    for run in eval_runs:
        run_id = run["id"]
        run_name = run["name"]
        prompt = run["prompt"]
        assertions = run["assertions"]
        skill_name = run.get("skill", "")

        print(f"\n{'='*60}")
        print(f"  Eval {run_id}: {run_name} (Skill: {skill_name})")
        print(f"{'='*60}")

        # --- With-Skill ---
        print("  [1/5] Generating With-Skill response...")
        skill_content = read_skill_content(skill_name)
        prompt_with_skill = f"System Instructions / Guidelines:\n{skill_content}\n\nUser Question:\n{prompt}"
        res_with_text = run_gemini_cli(prompt_with_skill)

        judge_prompt = build_judge_prompt(prompt, res_with_text, assertions)

        print("  [2/5] Judge A (auto/Flash)...")
        judge_a_output = run_gemini_cli(judge_prompt)
        judge_a_result = parse_judge_output(judge_a_output, assertions)

        print("  [3/5] Judge B (pro)...")
        judge_b_output = run_gemini_cli(judge_prompt, model="pro")
        judge_b_result = parse_judge_output(judge_b_output, assertions)

        print("  [4/5] Consensus merge...")
        consensus = consensus_merge(judge_a_result, judge_b_result, assertions)
        with_passed = sum(1 for a in consensus if a.get("passed", False))

        # --- Baseline (single auto judge) ---
        print("  [5/5] Baseline (auto only)...")
        res_baseline = run_gemini_cli(prompt)
        baseline_judge_output = run_gemini_cli(build_judge_prompt(prompt, res_baseline, assertions))
        baseline_result = parse_judge_output(baseline_judge_output, assertions)
        baseline_lookup = {a["name"]: a for a in baseline_result.get("assertions", [])}
        baseline_passed = sum(1 for a in baseline_result.get("assertions", []) if a.get("passed", False))

        # Align results
        final_with = []
        for orig in assertions:
            name = orig["name"]
            c = next((x for x in consensus if x["name"] == name), {"passed": False, "reason": "No evaluation"})
            final_with.append({
                "name": name,
                "description": orig["description"],
                "condition": orig["condition"],
                "passed": c.get("passed", False),
                "reason": c.get("reason", ""),
                "judge_a": c.get("judge_a"),
                "judge_b": c.get("judge_b"),
            })

        final_baseline = []
        for orig in assertions:
            name = orig["name"]
            b = baseline_lookup.get(name, {"passed": False, "reason": "No evaluation"})
            final_baseline.append({
                "name": name,
                "description": orig["description"],
                "condition": orig["condition"],
                "passed": b.get("passed", False),
                "reason": b.get("reason", ""),
            })

        pass_rate_with = (with_passed / len(assertions)) if assertions else 0.0
        pass_rate_base = (baseline_passed / len(assertions)) if assertions else 0.0
        print(f"  → With-Skill: {with_passed}/{len(assertions)} ({pass_rate_with:.0%}) | Baseline: {baseline_passed}/{len(assertions)} ({pass_rate_base:.0%})")

        results.append({
            "id": run_id,
            "name": run_name,
            "skill": skill_name,
            "prompt": prompt,
            "with_skill": {
                "success": True,
                "response": res_with_text[:500],  # Truncate for storage
                "assertions": final_with,
                "passed_count": with_passed,
                "total_assertions": len(assertions),
                "pass_rate": pass_rate_with,
            },
            "baseline": {
                "success": True,
                "response": res_baseline[:500],
                "assertions": final_baseline,
                "passed_count": baseline_passed,
                "total_assertions": len(assertions),
                "pass_rate": pass_rate_base,
            }
        })

    # Summary
    total_evals = len(results)
    mean_with = sum(r["with_skill"]["pass_rate"] for r in results) / total_evals if total_evals else 0.0
    mean_base = sum(r["baseline"]["pass_rate"] for r in results) / total_evals if total_evals else 0.0

    output_data = {
        "results": results,
        "summary": {
            "total_evals": total_evals,
            "judge_mode": "dual_consensus_auto_pro",
            "with_skill": {"mean_pass_rate": mean_with},
            "baseline": {"mean_pass_rate": mean_base},
            "pass_rate_improvement": mean_with - mean_base,
        }
    }

    grading_path = "evals/grading.json"
    os.makedirs(os.path.dirname(grading_path), exist_ok=True)
    with open(grading_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    with open("grading.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  DUAL-JUDGE CONSENSUS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total Eval Cases:        {total_evals}")
    print(f"  With-Skill Pass Rate:    {mean_with:.2%}")
    print(f"  Baseline Pass Rate:      {mean_base:.2%}")
    print(f"  Improvement Delta:       {mean_with - mean_base:+.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
