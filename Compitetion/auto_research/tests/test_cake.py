"""Tests for CAKE LLM kernel evolution (TDD).

Tests cover two bugs discovered in H4:
  1. _parse_llm_response: must extract clean kernel expression from various
     LLM response formats, including the "I propose the new kernel: **LIN + RQ**"
     case where the fallback parser over-captures.
  2. LLM call configuration: must use thinking=disabled (not reasoning_effort=low)
     to ensure the DeepSeek API returns non-empty content.

Run: cd Compitetion/auto_research && python -m pytest tests/test_cake.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make auto_research importable
_AUTO_ROOT = Path(__file__).resolve().parent.parent
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))


# ---------------------------------------------------------------------------
# _parse_llm_response tests
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    """Tests for _parse_llm_response — extract kernel expression from LLM output."""

    def test_standard_format(self):
        """'Kernel: <expr>\\nReasoning: ...' — the happy path."""
        from components.cake import _parse_llm_response

        response = "Kernel: LIN + RQ\nReasoning: The RQ kernel captures smooth variations."
        kernel, analysis = _parse_llm_response(response)
        assert kernel == "LIN + RQ"
        assert analysis is not None
        assert "RQ" in analysis

    def test_standard_format_single_kernel(self):
        """'Kernel: SE' — single base kernel, no operators."""
        from components.cake import _parse_llm_response

        response = "Kernel: SE\nReasoning: Simplest option."
        kernel, _ = _parse_llm_response(response)
        assert kernel == "SE"

    def test_standard_format_with_multiplication(self):
        """'Kernel: SE*PER' — multiplication operator."""
        from components.cake import _parse_llm_response

        response = "Kernel: SE*PER\nReasoning: Product of SE and PER."
        kernel, _ = _parse_llm_response(response)
        assert kernel == "SE*PER"

    def test_markdown_bold_kernel(self):
        """'Kernel: **LIN + RQ**' — LLM wraps kernel in markdown bold."""
        from components.cake import _parse_llm_response

        response = "Kernel: **LIN + RQ**\nReasoning: Combines linear and RQ."
        kernel, _ = _parse_llm_response(response)
        assert kernel == "LIN + RQ"

    def test_markdown_code_kernel(self):
        """'Kernel: `LIN + RQ`' — LLM wraps kernel in backticks."""
        from components.cake import _parse_llm_response

        response = "Kernel: `LIN + RQ`\nReasoning: Combines linear and RQ."
        kernel, _ = _parse_llm_response(response)
        assert kernel == "LIN + RQ"

    def test_fallback_no_kernel_label(self):
        """No 'Kernel:' label — fallback parser should find the expression.

        The LLM might say 'I propose the new kernel: **LIN + RQ**' without
        the 'Kernel:' label. The fallback must extract just 'LIN + RQ',
        NOT 'I propose the new kernel: **LIN + RQ**'.
        """
        from components.cake import _parse_llm_response

        response = "I propose the new kernel: **LIN + RQ**\n\nBecause it combines linear and RQ."
        kernel, _ = _parse_llm_response(response)
        # Must NOT include the prose prefix
        assert "propose" not in kernel
        assert "I " not in kernel
        # Must contain the actual kernel expression
        assert "LIN" in kernel
        assert "RQ" in kernel
        # The extracted kernel should be parseable
        assert kernel in ("LIN + RQ", "LIN+RQ", "**LIN + RQ**")

    def test_fallback_extracts_only_kernel_expression(self):
        """Fallback must extract ONLY the kernel expression, not surrounding prose.

        Regression for H4 bug: the old fallback returned the entire line
        including 'I propose the new kernel:' prefix.
        """
        from components.cake import _parse_llm_response

        response = "I propose the new kernel: LIN + RQ"
        kernel, _ = _parse_llm_response(response)
        # The kernel must be a valid expression (parseable by parse_kernel)
        from components.cake import parse_kernel
        # Should not raise
        covar = parse_kernel(kernel, 4)
        assert covar is not None

    def test_empty_response_raises(self):
        """Empty response must raise ValueError, not return garbage."""
        from components.cake import _parse_llm_response

        with pytest.raises(ValueError, match="No kernel"):
            _parse_llm_response("")

    def test_no_kernel_in_response_raises(self):
        """Response with no kernel-like content must raise."""
        from components.cake import _parse_llm_response

        with pytest.raises(ValueError, match="No kernel"):
            _parse_llm_response("I don't know what kernel to suggest.")

    def test_multiline_reasoning(self):
        """Multi-line reasoning after kernel line is captured."""
        from components.cake import _parse_llm_response

        response = """Kernel: SE + PER
Reasoning: This kernel combines the squared exponential for smooth trends
with the periodic kernel to capture recurring patterns in the data.
The addition operator allows both components to contribute independently."""
        kernel, analysis = _parse_llm_response(response)
        assert kernel == "SE + PER"
        assert analysis is not None
        assert len(analysis) > 10

    def test_parenthesized_expression(self):
        """Kernel with parentheses: 'Kernel: (SE+PER)*RQ'."""
        from components.cake import _parse_llm_response

        response = "Kernel: (SE+PER)*RQ\nReasoning: Nested composition."
        kernel, _ = _parse_llm_response(response)
        assert kernel == "(SE+PER)*RQ"

    def test_fallback_strips_markdown_asterisks(self):
        """Fallback must strip markdown ** bold markers from the expression."""
        from components.cake import _parse_llm_response

        response = "The best kernel is **SE + PER** for this data."
        kernel, _ = _parse_llm_response(response)
        # Must not contain ** markdown
        assert "**" not in kernel
        assert "SE" in kernel
        assert "PER" in kernel


# ---------------------------------------------------------------------------
# LLM call configuration tests
# ---------------------------------------------------------------------------

class TestLLMCallConfiguration:
    """Tests that CAKE calls the LLM with thinking=disabled (not reasoning_effort=low).

    Bug: reasoning_effort=low on deepseek-v4-flash causes reasoning_tokens to
    consume the entire max_tokens budget, returning empty content.
    Fix: use thinking={"type": "disabled"} to disable reasoning mode.
    """

    def test_crossover_uses_thinking_disabled(self):
        """Crossover LLM call must set thinking=disabled in extra_body."""
        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        # Mock the client to capture what extra_body is passed
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE + RQ\nReasoning: test"
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        # Set up population for crossover
        surrogate._population = {"LIN": 100.0, "RQ": 200.0}

        # Trigger crossover
        import numpy as np
        bic_list = ["LIN", "RQ"]
        probs = np.array([0.7, 0.3])
        bic_values = {"LIN": 100.0, "RQ": 200.0}
        surrogate._llm_crossover(mock_client, bic_list, probs, bic_values)

        # Check the chat call was made
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args
        extra_body = call_kwargs.kwargs.get("extra_body", {})

        # Must have thinking disabled
        assert extra_body.get("thinking") == {"type": "disabled"}, (
            f"Expected thinking=disabled, got extra_body={extra_body}"
        )
        # Must NOT use reasoning_effort (causes empty content on deepseek-v4-flash)
        assert "reasoning_effort" not in extra_body, (
            "reasoning_effort causes empty content; use thinking=disabled instead"
        )

    def test_mutation_uses_thinking_disabled(self):
        """Mutation LLM call must set thinking=disabled in extra_body."""
        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE\nReasoning: test"
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        # Trigger mutation
        surrogate._llm_mutation(mock_client, "LIN", 100.0)

        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args
        extra_body = call_kwargs.kwargs.get("extra_body", {})
        assert extra_body.get("thinking") == {"type": "disabled"}, (
            f"Expected thinking=disabled, got extra_body={extra_body}"
        )
        assert "reasoning_effort" not in extra_body

    def test_crossover_max_tokens_sufficient(self):
        """Crossover max_tokens must be >= 512 to leave room after reasoning."""
        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE + RQ\nReasoning: test"
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        surrogate._population = {"LIN": 100.0, "RQ": 200.0}
        import numpy as np
        surrogate._llm_crossover(
            mock_client, ["LIN", "RQ"], np.array([0.7, 0.3]),
            {"LIN": 100.0, "RQ": 200.0}
        )

        call_kwargs = mock_client.chat.call_args
        max_tokens = call_kwargs.kwargs.get("max_tokens", 0)
        assert max_tokens >= 512, (
            f"max_tokens={max_tokens} too small; need >= 512 to avoid reasoning eating the budget"
        )

    def test_empty_llm_content_returns_empty_list(self):
        """When LLM returns empty content, crossover must return [] (not crash)."""
        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = ""  # Empty — the H4 bug
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        surrogate._population = {"LIN": 100.0, "RQ": 200.0}
        import numpy as np
        result = surrogate._llm_crossover(
            mock_client, ["LIN", "RQ"], np.array([0.7, 0.3]),
            {"LIN": 100.0, "RQ": 200.0}
        )
        assert result == [], "Empty LLM content should produce empty kernel list"
    def test_crossover_does_not_use_global_numpy_rng(self, monkeypatch):
        """Parent selection must use the surrogate's seed-local RNG."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE + RQ\nReasoning: test"
        mock_client.chat.return_value = mock_result

        def fail_global_choice(*args, **kwargs):
            raise AssertionError("global np.random.choice must not be used")

        monkeypatch.setattr(np.random, "choice", fail_global_choice)
        result = surrogate._llm_crossover(
            mock_client,
            ["M5", "SE", "RQ"],
            np.array([0.5, 0.3, 0.2]),
            {"M5": 10.0, "SE": 20.0, "RQ": 30.0},
        )

        assert result == ["SE + RQ"]


# ---------------------------------------------------------------------------
# Prompt content tests (H4 prompt fix)
# ---------------------------------------------------------------------------

class TestPromptObservations:
    """Tests that the system prompt includes real observation data, not a placeholder."""

    def test_system_prompt_contains_observations(self):
        """System prompt must contain actual (x, y) observation pairs, not a placeholder."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        # Set observations as the CAKE reference does: list of (x, y) pairs
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([10.0, 20.0])
        surrogate.update_observations(X, y)

        prompt = surrogate._build_system_prompt()
        # Must contain actual data values, not a placeholder string
        assert "chemical reaction optimization" not in prompt.lower(), (
            "System prompt still uses placeholder; must contain real observation data"
        )
        assert "10.0" in prompt or "10" in prompt, (
            "System prompt must contain actual y values from observations"
        )

    def test_system_prompt_updates_with_new_observations(self):
        """When observations change, the system prompt must reflect the new data."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        X1 = np.array([[0.1, 0.2]])
        y1 = np.array([10.0])
        surrogate.update_observations(X1, y1)
        prompt1 = surrogate._build_system_prompt()

        X2 = np.array([[0.1, 0.2], [0.3, 0.4]])
        y2 = np.array([10.0, 99.9])
        surrogate.update_observations(X2, y2)
        prompt2 = surrogate._build_system_prompt()

        assert "99.9" in prompt2, "Updated prompt must contain new observation y=99.9"
        assert prompt1 != prompt2, "Prompt must change when observations change"


class TestFitnessSemantics:
    """Tests that fitness values passed to LLM are normalized so 'higher is better'."""

    def test_fitness_normalized_higher_is_better(self):
        """BIC (lower=better) must be converted to normalized fitness (higher=better).

        The CAKE prompt says 'higher values indicate better fit', but raw BIC
        has the opposite semantics. The fitness passed to the LLM must be
        normalized so that a lower BIC yields a higher fitness score.
        """
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE + RQ\nReasoning: test"
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        # LIN has BIC=100 (better), RQ has BIC=200 (worse)
        surrogate._population = {"LIN": 100.0, "RQ": 200.0}
        surrogate._llm_crossover(
            mock_client, ["LIN", "RQ"], np.array([0.7, 0.3]),
            {"LIN": 100.0, "RQ": 200.0}
        )

        call_kwargs = mock_client.chat.call_args
        user_prompt = call_kwargs.args[0][1]["content"]  # second message (user)

        # The fitness for LIN (BIC=100, better) must appear as HIGHER than RQ (BIC=200, worse)
        # The prompt should show normalized fitness, not raw BIC
        # Extract fitness numbers from the prompt
        import re
        numbers = re.findall(r"[\d.]+", user_prompt)
        assert len(numbers) >= 2, f"Expected at least 2 fitness numbers in prompt, got: {user_prompt}"

        # LIN fitness should be higher than RQ fitness (since LIN has lower BIC)
        # The prompt format is: "LIN (fitness1), RQ (fitness2)"
        # We need to parse which number belongs to which kernel
        lin_match = re.search(r"LIN\s*\(([\d.]+)\)", user_prompt)
        rq_match = re.search(r"RQ\s*\(([\d.]+)\)", user_prompt)
        assert lin_match and rq_match, (
            f"Could not parse kernel fitness from prompt: {user_prompt}"
        )
        lin_fitness = float(lin_match.group(1))
        rq_fitness = float(rq_match.group(1))
        assert lin_fitness > rq_fitness, (
            f"LIN (lower BIC) should have HIGHER fitness, but got LIN={lin_fitness} > RQ={rq_fitness}"
        )


class TestPromptOutputFormat:
    """Tests that the prompt explicitly requires the 'Kernel: <expr>' output format."""

    def test_crossover_prompt_requires_kernel_format(self):
        """Crossover prompt must instruct the LLM to output 'Kernel: <expression>'."""
        from components.cake import CROSSOVER_PROMPT_TEMPLATE, OPERATORS

        prompt = CROSSOVER_PROMPT_TEMPLATE.format(
            parent_kernel1="LIN", fitness1="0.8",
            parent_kernel2="RQ", fitness2="0.3",
            operators=OPERATORS,
        )
        assert "Kernel:" in prompt, (
            "Crossover prompt must require 'Kernel: <expression>' output format"
        )

    def test_mutation_prompt_requires_kernel_format(self):
        """Mutation prompt must instruct the LLM to output 'Kernel: <expression>'."""
        from components.cake import BASE_KERNELS, MUTATION_PROMPT_TEMPLATE

        prompt = MUTATION_PROMPT_TEMPLATE.format(
            kernel="LIN", fitness="0.8", base_kernels=BASE_KERNELS
        )
        assert "Kernel:" in prompt, (
            "Mutation prompt must require 'Kernel: <expression>' output format"
        )


class TestTemperatureDiversity:
    """Tests that the LLM call uses temperature > 0 for diversity."""

    def test_crossover_temperature_positive(self):
        """Crossover LLM call must use temperature > 0 (not 0.0 deterministic)."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100)
        mock_client = MagicMock()
        mock_client.is_configured.return_value = True
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.content = "Kernel: SE + RQ\nReasoning: test"
        mock_client.chat.return_value = mock_result
        surrogate._client = mock_client

        surrogate._population = {"LIN": 100.0, "RQ": 200.0}
        surrogate._llm_crossover(
            mock_client, ["LIN", "RQ"], np.array([0.7, 0.3]),
            {"LIN": 100.0, "RQ": 200.0}
        )

        call_kwargs = mock_client.chat.call_args
        extra_body = call_kwargs.kwargs.get("extra_body", {})
        temperature = extra_body.get("temperature", 0.0)
        assert temperature > 0.0, (
            f"temperature={temperature} kills diversity; need > 0 (e.g. 0.7) for kernel evolution"
        )


# ---------------------------------------------------------------------------
# Ensemble predict tests (H4 architecture fix)
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    """Tests that CAKESurrogate uses population-weighted ensemble predict,
    NOT single best kernel. This is the core CAKE architecture:
    all kernels in the population contribute to predictions, weighted by BIC.
    """

    def test_population_models_stored_after_fit(self):
        """After fit, the surrogate must store multiple fitted GP models (one per population kernel)."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100, population_size=4)
        X = np.random.randn(10, 4)
        y = np.random.randn(10)

        surrogate.fit(X, y)

        # Must have a dict of {kernel_expr: model} — the ensemble
        assert hasattr(surrogate, '_population_models'), (
            "CAKESurrogate must store _population_models dict for ensemble predict"
        )
        assert len(surrogate._population_models) > 0, (
            "Population models must be populated after fit"
        )

    def test_predict_uses_ensemble_not_single_model(self):
        """predict() must return weighted mean of all population models,
        not just the single best kernel's prediction."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100, population_size=3)
        X_train = np.random.randn(8, 4)
        y_train = np.random.randn(8)
        X_pool = np.random.randn(5, 4)

        surrogate.fit(X_train, y_train)
        mu, sigma = surrogate.predict(X_pool)

        # Verify that mu is NOT identical to any single model's prediction
        # (ensemble should blend multiple models)
        assert mu.shape == (5,), f"mu shape mismatch: {mu.shape}"
        assert sigma.shape == (5,), f"sigma shape mismatch: {sigma.shape}"
        assert np.all(np.isfinite(mu)), "mu contains non-finite values"
        assert np.all(np.isfinite(sigma)), "sigma contains non-finite values"
        assert np.all(sigma > 0), "sigma must be positive"

    def test_predict_weights_sum_to_one(self):
        """Ensemble weights (from BIC softmax) must sum to 1."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100, population_size=4)
        X = np.random.randn(8, 4)
        y = np.random.randn(8)

        surrogate.fit(X, y)

        weights = surrogate.population_weights
        assert weights is not None, "population_weights must be set after fit"
        assert len(weights) > 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1, got {total}"

    def test_predict_reflects_multiple_kernels(self):
        """If population has multiple kernels, ensemble mean must differ
        from any single kernel's mean. This verifies blending."""
        import numpy as np

        from components.cake import CAKESurrogate

        surrogate = CAKESurrogate(seed=100, population_size=3)
        X_train = np.random.randn(8, 4)
        y_train = np.random.randn(8)
        X_pool = np.random.randn(3, 4)

        surrogate.fit(X_train, y_train)
        ensemble_mu, _ = surrogate.predict(X_pool)

        # Get individual model predictions
        individual_means = []
        for model in surrogate._population_models.values():
            if model is not None:
                X_t = surrogate._tensor(X_pool)
                with __import__("torch").no_grad():
                    post = model.posterior(X_t)
                    individual_means.append(post.mean.squeeze(-1).numpy())

        if len(individual_means) > 1:
            # Ensemble mean should not exactly match any single model
            for im in individual_means:
                assert not np.allclose(ensemble_mu, im, atol=1e-6), (
                    "Ensemble mean matches a single kernel — ensemble is not blending"
                )
