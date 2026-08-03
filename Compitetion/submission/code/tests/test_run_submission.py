"""Tests for the competition submission entry point."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from bo_core import llm_client
from bo_core.llm_client import DeepSeekClient, LlmCallResult
from bo_core.optimization.chem_lgbo import ChemLGBOEngine
from main import run_submission


def test_main_runs_lgbo_with_40_query_budget(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run_one(
        dataset,
        method,
        seed,
        n_iters,
        output_dir,
        *,
        backend,
    ):
        calls.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "n_iters": n_iters,
                "output_dir": output_dir,
                "backend": backend,
            }
        )
        return {"best_found": 80.0}

    monkeypatch.setattr(run_submission, "SEEDS", [100])
    monkeypatch.setattr(run_submission, "DATASETS", ["buchwald_sub4"])
    monkeypatch.setattr(run_submission, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_submission, "run_one", fake_run_one)

    run_submission.main()

    assert calls == [
        {
            "dataset": "buchwald_sub4",
            "method": "lgbo",
            "seed": 100,
            "n_iters": 40,
            "output_dir": tmp_path,
            "backend": "botorch",
        }
    ]


def test_project_env_is_the_only_file_source_and_process_env_wins(
    monkeypatch, tmp_path: Path
) -> None:
    assert llm_client.PROJECT_ENV_PATH == Path(__file__).resolve().parents[4] / ".env"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "DEEPSEEK_API_KEY=file-key\nDEEPSEEK_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_client, "PROJECT_ENV_PATH", project_env)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    monkeypatch.delenv("DEEPSEEK_FLASH_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    client = DeepSeekClient.from_env()

    assert client.api_key == "process-key"
    assert client.model == "file-model"
    assert llm_client.PROJECT_ENV_PATH == project_env


def test_submission_tests_import_vendored_bo_core() -> None:
    assert Path(llm_client.__file__).resolve().is_relative_to(CODE_ROOT)


def test_vendored_client_accepts_empty_content_tool_call(
    monkeypatch,
) -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "propose_sparse_subspace",
                "arguments": '{"subspace":{"Ligand":["L1"]}}',
            },
        }
    ]

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "", "tool_calls": tool_calls}}]}

    monkeypatch.setattr(llm_client.requests, "post", lambda *_args, **_kwargs: Response())
    client = DeepSeekClient("test-key")

    result = client.chat([{"role": "user", "content": "propose"}], temperature=0.2)

    assert result.status == "success"
    assert result.content == ""
    assert result.tool_calls == tool_calls


def test_vendored_chem_accepts_scripted_tool_call() -> None:
    engine = ChemLGBOEngine(
        "suzuki",
        seed=100,
        use_llm=False,
        n_iters=0,
        n_restarts=0,
        backend="sklearn",
    )
    ligand = str(engine.test_df["Ligand"].iloc[0])

    class Client:
        def is_configured(self):
            return True
        def chat(self, _messages, **_kwargs):
            return LlmCallResult(
                status="success",
                provider="test",
                model="test",
                content="",
                usage={},
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "propose_sparse_subspace",
                            "arguments": (
                                '{"subspace":{"Ligand":["' + ligand + '"]}}'
                            ),
                        },
                    }
                ],
            )

    engine.use_llm = True
    engine._client = Client()
    engine.step()

    artifact = engine.guidance_artifacts[0]
    assert artifact["parser_reason"] == "accepted"
    assert artifact["tool_call_id"] == "call_1"
    assert engine.llm_temperature == 0.2
