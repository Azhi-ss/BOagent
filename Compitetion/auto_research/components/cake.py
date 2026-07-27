"""CAKE: Context-Aware Kernel Evolution (LLM-guided kernel evolution).

Reference: "Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE
with LLMs" (NeurIPS 2025, arXiv:2509.17998).

CAKE uses an LLM to evolve Gaussian Process kernel expressions via crossover
and mutation, guided by a BIC-based fitness function. This is the "LLM
intervention at kernel structure" — distinct from the existing LGBO mean-shift,
which intervenes at the posterior mean.

The evolution loop:
  1. Initialize population with diverse base kernels (SE, PER, LIN, RQ, M3, M5)
  2. Compute fitness (BIC) for each kernel on observed data
  3. LLM proposes new kernels via crossover (combine two parents) and
     mutation (replace a base kernel in one expression)
  4. Evaluate new kernels' BIC, keep top-N (selection)
  5. Repeat every ``evolve_interval`` BO iterations

Key difference from H1 (Kernel Manifold): CAKE uses the LLM to *generate*
kernel expressions, not just select from a fixed library. The LLM's chemical
reasoning can propose compositions a geometric search would miss.

Fallback: on any failure (LLM API, parse, fit), the surrogate falls back to
the submission's fixed Matern-5/2 kernel.
"""
from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_SUBMISSION_CODE = Path(__file__).resolve().parents[3] / "submission" / "code"
if str(_SUBMISSION_CODE) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_CODE))

from bo_core.optimization.surrogate import (
    _MIN_STD,
    LBFGSB_MAX_LINE_SEARCH_STEPS,
    BoTorchSurrogate,
    _validate_prediction,
)

# ---------------------------------------------------------------------------
# Kernel expression parsing (reused from kernel_manifold.py logic)
# ---------------------------------------------------------------------------

def _base_kernel_factory(name: str, d: int):
    from gpytorch.kernels import (
        LinearKernel,
        MaternKernel,
        PeriodicKernel,
        RBFKernel,
        RQKernel,
    )
    table = {
        "SE": lambda: RBFKernel(ard_num_dims=d),
        "PER": lambda: PeriodicKernel(ard_num_dims=d),
        "LIN": lambda: LinearKernel(ard_num_dims=d),
        "RQ": lambda: RQKernel(ard_num_dims=d),
        "M1": lambda: MaternKernel(nu=0.5, ard_num_dims=d),
        "M3": lambda: MaternKernel(nu=1.5, ard_num_dims=d),
        "M5": lambda: MaternKernel(nu=2.5, ard_num_dims=d),
    }
    if name not in table:
        raise ValueError(f"Unknown base kernel: {name!r}")
    return table[name]()


def parse_kernel(expression: str, d: int):
    """Parse a kernel expression like 'SE+PER' into a gpytorch ScaleKernel."""
    from gpytorch.kernels import ScaleKernel

    base_kernels: dict[str, Any] = {name: _base_kernel_factory(name, d) for name in
                                    ("SE", "PER", "LIN", "RQ", "M1", "M3", "M5")}

    def apply_op(left, op, right):
        if op == "+":
            return left + right
        if op == "*":
            return left * right
        raise ValueError(f"Unknown operator: {op!r}")

    def parse_subexpr(subexpr: str):
        names = re.findall(r"\w+", subexpr)
        ops = re.findall(r"[+*]", subexpr)
        if not names:
            raise ValueError(f"Empty subexpression: {subexpr!r}")
        if names[0] not in base_kernels:
            raise ValueError(f"Unknown kernel name: {names[0]!r}")
        result = base_kernels[names[0]]
        for i, op in enumerate(ops):
            nxt = names[i + 1]
            if nxt not in base_kernels:
                raise ValueError(f"Unknown kernel name: {nxt!r}")
            result = apply_op(result, op, base_kernels[nxt])
        return ScaleKernel(result)

    pattern = r"\(([^()]+)\)"
    cache: dict[str, Any] = {}
    while "(" in expression:
        for subexpr in re.findall(pattern, expression):
            if subexpr not in cache:
                sub_kernel = parse_subexpr(subexpr)
                cache[subexpr] = sub_kernel
                base_kernels[f"SubKernel{len(base_kernels)}"] = sub_kernel
            expression = expression.replace(f"({subexpr})", f"SubKernel{len(base_kernels) - 1}", 1)
    return parse_subexpr(expression)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are an expert in machine learning and chemistry, specializing in Gaussian processes for Bayesian optimization of chemical reactions. Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can be captured by a kernel function.
You can use any of the following base kernels: {base_kernels}, and combine these kernels using the following operators: {operators}.
Your goal is to construct a kernel expression that best explains the observed data.
The kernel will be evaluated using a fitness score normalized between [0, 1], where higher values indicate better fit to the data."""

CROSSOVER_PROMPT_TEMPLATE = """You are given two parent kernels and their fitness scores (higher is better):
{parent_kernel1} ({fitness1}),  {parent_kernel2} ({fitness2})

Please propose a new kernel that has a potentially higher fitness score.
You may combine the parent kernels using any of the operators from: {operators}.
Briefly explain your reasoning behind the proposed kernel.

You MUST respond in this exact format:
Kernel: <expression>
Reasoning: <explanation>"""

MUTATION_PROMPT_TEMPLATE = """You are given a kernel and its fitness score (higher is better):
{kernel} ({fitness})

Please propose a new kernel that has a potentially higher fitness score.
You may replace a base kernel in the current expression with another base kernel from the set: {base_kernels}.
Briefly explain your reasoning behind the proposed kernel.

You MUST respond in this exact format:
Kernel: <expression>
Reasoning: <explanation>"""

BASE_KERNELS = ["SE", "PER", "LIN", "RQ", "M1", "M3", "M5"]
OPERATORS = ["+", "*"]


def _bic_to_fitness(bic: float, all_bics: list[float]) -> float:
    """Convert BIC (lower is better) to normalized fitness in [0, 1] (higher is better).

    Uses softmax of (-BIC / scale) so lower BIC → higher fitness.
    The scale is the std of the BIC values for numerical stability.
    """
    arr = np.array(all_bics, dtype=float)
    if arr.std() > 1e-9:
        normalized = (arr - arr.mean()) / arr.std()
    else:
        normalized = np.zeros_like(arr)
    # softmax of -normalized → lower BIC gets higher probability
    probs = np.exp(-normalized - np.max(-normalized))
    probs = probs / probs.sum()
    idx = list(all_bics).index(bic)
    return float(probs[idx])


def _parse_llm_response(response: str) -> tuple[str, str | None]:
    """Parse the LLM response to extract the kernel expression and analysis.

    Expected format:
        Kernel: <expression>
        Reasoning: <analysis>

    Fallback for responses without "Kernel:" label: scan each line for a
    substring that looks like a valid kernel expression (composed of base
    kernel names and +/* operators) and extract ONLY that substring,
    stripping surrounding prose and markdown.

    Returns (kernel_expression, analysis_text). On parse failure, raises ValueError.
    """
    # Build a regex that matches a kernel expression: one or more base
    # kernel names joined by + or *, optionally parenthesized.
    base_alt = "|".join(BASE_KERNELS)
    # A single base kernel: M5, SE, etc.
    single = rf"(?:{base_alt})"
    # Composed: SE+PER, (SE+PER)*RQ, etc. Allow parentheses.
    # Must start and end with a base kernel name (or closing paren).
    kernel_re = re.compile(
        rf"({single}(?:\s*[+*]\s*(?:\({single}(?:\s*[+*]\s*{single})*\)|{single}))*)"
    )

    # Primary path: look for "Kernel:" label
    kernel_start = response.find("Kernel:")
    if kernel_start != -1:
        kernel_start += len("Kernel:")
        kernel_end = response.find("\n", kernel_start)
        if kernel_end == -1:
            kernel_end = len(response)
        kernel = response[kernel_start:kernel_end].strip()
        # Strip markdown/code formatting
        kernel = re.sub(r"`+", "", kernel)
        kernel = re.sub(r"^\*+|\*+$", "", kernel).strip()
        if kernel:
            analysis = response[kernel_end:].strip()[:500] or None
            return kernel, analysis

    # Fallback: scan for kernel-expression substrings anywhere in the response.
    # Pick the LONGEST match (most specific expression), not the first —
    # the LLM may mention base kernel names in prose before stating the
    # actual composite expression (e.g., "LIN is better than RQ ... LIN * RQ").
    matches = list(kernel_re.finditer(response))
    if matches:
        longest = max(matches, key=lambda m: len(m.group(1)))
        kernel = longest.group(1).strip()
        # Strip markdown bold/code wrapping if present
        kernel = re.sub(r"^\*+|\*+$", "", kernel).strip()
        kernel = re.sub(r"`+", "", kernel).strip()
        if kernel:
            analysis = response[:500] or None
            return kernel, analysis

    raise ValueError(f"No kernel expression found in response: {response[:200]}")


# ---------------------------------------------------------------------------
# BIC fitness computation
# ---------------------------------------------------------------------------

def _compute_bic(train_X, train_Y, kernel_expr: str, dimension: int, max_iter: int = 50) -> float:
    """Compute BIC for a kernel expression on the training data.

    BIC = -2 * log_likelihood + num_params * log(num_data)
    Lower BIC is better. Returns +inf on failure.
    """
    import gpytorch
    import torch
    from botorch.fit import fit_gpytorch_mll_scipy
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    try:
        covar_module = parse_kernel(kernel_expr, dimension)
        model = SingleTaskGP(
            train_X, train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=dimension),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.cholesky_jitter(double_value=1e-2):
            fit_gpytorch_mll_scipy(mll, options={"maxiter": max_iter, "maxls": LBFGSB_MAX_LINE_SEARCH_STEPS})
        model.eval()
        with torch.no_grad():
            output = model(train_X)
            log_likelihood = mll(output, train_Y.squeeze(-1)).item()
        num_params = sum(p.numel() for p in model.parameters())
        num_data = train_X.size(0)
        bic = -2.0 * log_likelihood + num_params * math.log(max(num_data, 1))
        return float(bic)
    except Exception:  # noqa: BLE001 - BIC computation is best-effort
        return float("inf")


# ---------------------------------------------------------------------------
# CAKESurrogate
# ---------------------------------------------------------------------------

class CAKESurrogate(BoTorchSurrogate):
    """BoTorchSurrogate subclass that evolves kernels via LLM (CAKE).

    Every ``evolve_interval`` fits, the LLM is queried to propose new kernel
    expressions via crossover and mutation. Each proposal is evaluated by BIC
    on the current observed data; the population is pruned to the top-N
    fittest kernels. The best kernel is used for the subsequent GP fit.

    On any failure (LLM API, parse, BIC), the surrogate falls back to
    Matern-5/2 (matching the submission's fixed kernel behavior).
    """

    def __init__(
        self,
        *,
        seed: int,
        alpha: float = 1e-4,
        jitter_levels: Sequence[float] | None = None,
        max_fit_iterations: int = 100,
        population_size: int = 6,
        evolve_interval: int = 5,
        selection_max_iter: int = 50,
        chat_engine: str = "deepseek-v4-flash",
        reasoning_effort: str = "low",
    ) -> None:
        super().__init__(
            seed=seed,
            alpha=alpha,
            jitter_levels=jitter_levels,
            max_fit_iterations=max_fit_iterations,
        )
        self.population_size = int(population_size)
        self.evolve_interval = max(1, int(evolve_interval))
        self.selection_max_iter = int(selection_max_iter)
        self.chat_engine = chat_engine
        self.reasoning_effort = reasoning_effort
        self._rng = np.random.RandomState(seed)

        self._fit_count = 0
        self._current_kernel = "M5"
        # population: {kernel_expr: bic}; lower BIC is better
        self._population: dict[str, float] = {k: float("inf") for k in BASE_KERNELS}
        self._kernel_history: list[tuple[int, str, float]] = []
        self._llm_calls: int = 0
        self._client = None  # lazy init
        # Observations (X, y) for the system prompt; updated each fit
        self._obs_X: np.ndarray | None = None
        self._obs_y: np.ndarray | None = None
        # Ensemble state: one fitted GP model per population kernel
        self._population_models: dict[str, Any] = {}
        self._population_weights: dict[str, float] = {}

    def update_observations(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store observations for the LLM system prompt.

        Mirrors CAKE reference update_data(): the system prompt includes
        real (x, y) pairs so the LLM can reason about data patterns.
        """
        self._obs_X = np.asarray(X, dtype=float)
        self._obs_y = np.asarray(y, dtype=float)

    def _get_client(self):
        if self._client is None:
            from bo_core.llm_client import DeepSeekClient
            self._client = DeepSeekClient.from_env()
            self._client.model = self.chat_engine
            self._client.timeout_s = 60
        return self._client

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble of GPs (one per population kernel) and compute BIC weights.

        Unlike the single-kernel fit, this fits ALL kernels in the population
        and stores them for ensemble predict(). BIC is used only to compute
        softmax weights, NOT to select a single winner.
        """
        self._fit_count += 1
        train_X = self._tensor(X)
        train_Y = self._tensor(np.asarray(y, dtype=float).reshape(-1, 1))
        dimension = train_X.shape[-1]
        self.model = None

        # Update observations for the LLM system prompt
        self.update_observations(X, y)

        # Trigger kernel evolution on the first fit and every `evolve_interval`
        if self._fit_count == 1 or self._fit_count % self.evolve_interval == 1:
            try:
                self._evolve_kernel(train_X, train_Y, dimension)
            except Exception as exc:  # noqa: BLE001 - kernel evolution is best-effort
                print(f"[CAKESurrogate] kernel evolution failed: {exc}; using M5")
                self._current_kernel = "M5"

        # Fit ALL population kernels → ensemble
        self._population_models = {}
        bics: dict[str, float] = {}
        for kexpr in self._population:
            try:
                model, bic = self._fit_one_kernel(train_X, train_Y, kexpr, dimension)
                self._population_models[kexpr] = model
                bics[kexpr] = bic
            except Exception as exc:  # noqa: BLE001 - per-kernel fit is best-effort
                print(f"[CAKESurrogate] fit {kexpr} failed: {exc}; skipping")

        if not self._population_models:
            # Fallback: single Matern-5/2
            model, _ = self._fit_one_kernel(train_X, train_Y, "M5", dimension)
            self._population_models = {"M5": model}
            bics = {"M5": 1.0}

        # Compute softmax weights from BIC (lower BIC → higher weight)
        bic_arr = np.array(list(bics.values()))
        if bic_arr.std() > 1e-9:
            bic_norm = (bic_arr - bic_arr.mean()) / bic_arr.std()
        else:
            bic_norm = np.zeros_like(bic_arr)
        probs = np.exp(-bic_norm - np.max(-bic_norm))
        probs = probs / probs.sum()
        self._population_weights = {k: float(p) for k, p in zip(bics.keys(), probs)}

        # Set self.model to the highest-weight kernel (for backward-compat
        # with posterior_covariance calls in mean-shift)
        best_k = max(self._population_weights, key=self._population_weights.get)
        self.model = self._population_models[best_k]
        self._current_kernel = best_k
        self._inference_jitter = float(self.jitter_levels[0])
        print(
            f"[CAKESurrogate] ensemble fit#{self._fit_count}: "
            f"{len(self._population_models)} models, best={best_k} "
            f"(w={self._population_weights[best_k]:.3f})"
        )
        return self

    def _fit_one_kernel(self, train_X, train_Y, kexpr: str, dimension: int):
        """Fit a single GP with the given kernel expression; return (model, BIC)."""
        import gpytorch
        import torch
        from botorch.fit import fit_gpytorch_mll_scipy
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from linear_operator.utils.errors import NotPSDError

        try:
            covar_module = parse_kernel(kexpr, dimension)
        except Exception:  # noqa: BLE001 - parse fallback
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension))

        model = SingleTaskGP(
            train_X, train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=dimension),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        last_exc: Exception | None = None
        for jitter in self.jitter_levels:
            try:
                with gpytorch.settings.cholesky_jitter(double_value=float(jitter)):
                    fit_gpytorch_mll_scipy(
                        mll,
                        options={
                            "maxiter": self.max_fit_iterations,
                            "maxls": LBFGSB_MAX_LINE_SEARCH_STEPS,
                        },
                    )
                self._inference_jitter = float(jitter)
                last_exc = None
                break
            except NotPSDError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc

        model.eval()
        # Compute BIC
        with torch.no_grad():
            output = model(train_X)
            log_likelihood = mll(output, train_Y.squeeze(-1)).item()
        num_params = sum(p.numel() for p in model.parameters())
        num_data = train_X.size(0)
        bic = -2.0 * log_likelihood + num_params * math.log(max(num_data, 1))
        return model, float(bic)

    # -------------------------------------------------------------- evolution

    def _evolve_kernel(self, train_X, train_Y, dimension: int) -> None:
        """Run one CAKE evolution step: compute fitness, LLM crossover+mutation, select."""
        # Step 1: Compute BIC fitness for the current population
        bic_values: dict[str, float] = {}
        for kexpr in list(self._population.keys()):
            bic = _compute_bic(train_X, train_Y, kexpr, dimension, self.selection_max_iter)
            if math.isfinite(bic):
                bic_values[kexpr] = bic
        if not bic_values:
            self._current_kernel = "M5"
            return

        # Standardize fitness for selection probabilities
        bic_arr = np.array(list(bic_values.values()))
        if bic_arr.std() > 1e-9:
            bic_norm = (bic_arr - bic_arr.mean()) / bic_arr.std()
        else:
            bic_norm = np.zeros_like(bic_arr)
        # Lower BIC is better → higher prob. Use softmax of -bic_norm.
        probs = np.exp(-bic_norm - np.max(-bic_norm))
        probs = probs / probs.sum()
        bic_list = list(bic_values.keys())

        # Step 2: LLM crossover — propose new kernels by combining parents
        client = self._get_client()
        if client.is_configured():
            new_kernels = self._llm_crossover(client, bic_list, probs, bic_values)
            for kexpr in new_kernels:
                if kexpr not in bic_values and len(kexpr) < 30:
                    bic = _compute_bic(train_X, train_Y, kexpr, dimension, self.selection_max_iter)
                    if math.isfinite(bic):
                        bic_values[kexpr] = bic

            # Step 3: LLM mutation — replace a base kernel in the best expression
            best_kernel = min(bic_values, key=bic_values.get)
            mutated = self._llm_mutation(client, best_kernel, bic_values[best_kernel], list(bic_values.values()))
            for kexpr in mutated:
                if kexpr not in bic_values and len(kexpr) < 30:
                    bic = _compute_bic(train_X, train_Y, kexpr, dimension, self.selection_max_iter)
                    if math.isfinite(bic):
                        bic_values[kexpr] = bic

        # Step 4: Selection — keep top-N fittest kernels
        sorted_kernels = sorted(bic_values.items(), key=lambda x: x[1])  # ascending BIC
        self._population = dict(sorted_kernels[: self.population_size])

        # Step 5: Pick the best kernel
        best_kernel = min(bic_values, key=bic_values.get)
        best_bic = bic_values[best_kernel]
        prev = self._current_kernel
        self._current_kernel = best_kernel
        self._kernel_history.append((self._fit_count, best_kernel, best_bic))
        print(
            f"[CAKESurrogate] evolve@fit#{self._fit_count}: {prev} -> {best_kernel} "
            f"(BIC={best_bic:.2f}; population={len(self._population)})"
        )

    def _llm_crossover(
        self, client, bic_list: list[str], probs: np.ndarray, bic_values: dict[str, float]
    ) -> list[str]:
        """Query the LLM to propose kernels by crossing over two parents.

        Passes normalized fitness (higher=better) to the LLM, not raw BIC,
        so the prompt semantics ("higher is better") match the values.
        """
        if len(bic_list) < 2:
            return []
        # Pick two parents (weighted by fitness)
        idx = self._rng.choice(len(bic_list), size=min(2, len(bic_list)), p=probs, replace=False)
        p1, p2 = bic_list[idx[0]], bic_list[idx[1]]
        # Convert BIC to normalized fitness so "higher is better" is true
        all_bics = list(bic_values.values())
        f1 = _bic_to_fitness(bic_values[p1], all_bics)
        f2 = _bic_to_fitness(bic_values[p2], all_bics)
        prompt = CROSSOVER_PROMPT_TEMPLATE.format(
            parent_kernel1=p1, fitness1=f"{f1:.4f}",
            parent_kernel2=p2, fitness2=f"{f2:.4f}",
            operators=OPERATORS,
        )
        system_prompt = self._build_system_prompt()
        try:
            result = client.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}],
                max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}, "temperature": 0.7},
            )
            self._llm_calls += 1
            if getattr(result, "status", None) != "success" or not result.content:
                return []
            kernel, _ = _parse_llm_response(result.content)
            return [kernel]
        except Exception as exc:  # noqa: BLE001 - LLM is best-effort
            print(f"[CAKESurrogate] crossover LLM call failed: {exc}")
            return []

    def _llm_mutation(self, client, kernel: str, bic: float, all_bics: list[float] | None = None) -> list[str]:
        """Query the LLM to mutate the best kernel by replacing a base kernel.

        Passes normalized fitness (higher=better) to the LLM, not raw BIC.
        """
        if all_bics is None:
            all_bics = [bic]
        fitness = _bic_to_fitness(bic, all_bics)
        prompt = MUTATION_PROMPT_TEMPLATE.format(
            kernel=kernel, fitness=f"{fitness:.4f}", base_kernels=BASE_KERNELS
        )
        system_prompt = self._build_system_prompt()
        try:
            result = client.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}],
                max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}, "temperature": 0.7},
            )
            self._llm_calls += 1
            if getattr(result, "status", None) != "success" or not result.content:
                return []
            kernel, _ = _parse_llm_response(result.content)
            return [kernel]
        except Exception as exc:  # noqa: BLE001 - LLM is best-effort
            print(f"[CAKESurrogate] mutation LLM call failed: {exc}")
            return []

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current observation data.

        Mirrors CAKE reference update_data(): includes real (x, y) pairs
        so the LLM can reason about data patterns, not just kernel names.
        """
        if self._obs_X is not None and self._obs_y is not None:
            # Format observations as "x = [...], y = value" lines (CAKE reference format)
            lines = []
            for i in range(min(len(self._obs_X), 20)):  # cap at 20 for token budget
                x_str = ", ".join(f"{v:.2f}" for v in self._obs_X[i])
                lines.append(f"x = [{x_str}], y = {self._obs_y[i]:.2f}")
            observations = "\n".join(lines)
        else:
            observations = "(no observations yet)"
        return SYSTEM_PROMPT_TEMPLATE.format(
            observations=observations,
            base_kernels=BASE_KERNELS,
            operators=OPERATORS,
        )

    # -------------------------------------------------------------- ensemble predict

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Population-weighted ensemble prediction.

        For each kernel in the population, query its posterior mean and
        variance. Blend means and variances by BIC-derived weights.
        This mirrors the CAKE reference get_next_query() approach: all
        kernels contribute, weighted by fitness — no single winner.
        """
        import gpytorch
        import torch

        X_array = np.asarray(X, dtype=float)
        if not self._population_models:
            # Fallback: no ensemble (shouldn't happen after fit)
            return super().predict(X)

        weights = self._population_weights
        total_mean = np.zeros(len(X_array), dtype=float)
        total_var = np.zeros(len(X_array), dtype=float)
        weight_sum = 0.0

        with (
            torch.no_grad(),
            gpytorch.settings.cholesky_jitter(double_value=self._inference_jitter),
        ):
            for kexpr, model in self._population_models.items():
                w = weights.get(kexpr, 0.0)
                if w <= 0 or model is None:
                    continue
                means = []
                variances = []
                for start in range(0, len(X_array), self._PREDICTION_BATCH_SIZE):
                    X_block = X_array[start : start + self._PREDICTION_BATCH_SIZE]
                    posterior = model.posterior(self._tensor(X_block))
                    means.append(posterior.mean.squeeze(-1).detach().cpu().numpy())
                    variances.append(posterior.variance.squeeze(-1).detach().cpu().numpy())
                if not means:
                    continue
                k_mean = np.concatenate(means)
                k_var = np.concatenate(variances)
                # Weighted mixture of means
                total_mean += w * k_mean
                # Law of total variance: E[var] + Var[E]
                total_var += w * (k_var + k_mean**2)
                weight_sum += w

        if weight_sum > 0:
            total_mean /= weight_sum
            total_var = total_var / weight_sum - total_mean**2
            total_var = np.maximum(total_var, _MIN_STD**2)
        else:
            return super().predict(X)

        sigma = np.sqrt(total_var)
        return _validate_prediction(total_mean, sigma)

    # -------------------------------------------------------------- accessors

    @property
    def current_kernel(self) -> str:
        return self._current_kernel

    @property
    def kernel_history(self) -> list[tuple[int, str, float]]:
        return list(self._kernel_history)

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def population(self) -> dict[str, float]:
        return dict(self._population)

    @property
    def population_weights(self) -> dict[str, float] | None:
        return dict(self._population_weights) if self._population_weights else None


# ---------------------------------------------------------------------------
# Factory for the component registry
# ---------------------------------------------------------------------------

def create_cake_surrogate(
    backend: str,
    seed: int,
    **kwargs: Any,
) -> CAKESurrogate:
    """Factory matching the SURROGATES registry signature."""
    if backend != "botorch":
        raise ValueError(f"CAKESurrogate only supports 'botorch' backend, got {backend!r}")
    return CAKESurrogate(
        seed=seed,
        alpha=kwargs.get("alpha", 1e-2),
        population_size=kwargs.get("population_size", 6),
        evolve_interval=kwargs.get("evolve_interval", 5),
        selection_max_iter=kwargs.get("selection_max_iter", 50),
        chat_engine=kwargs.get("chat_engine", "deepseek-v4-flash"),
        reasoning_effort=kwargs.get("reasoning_effort", "low"),
    )
