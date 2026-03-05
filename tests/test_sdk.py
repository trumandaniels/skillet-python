from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from skillet import (
    ApiError,
    AsyncClient,
    AuthenticationError,
    AuthorizationError,
    BenchmarkTask,
    BuildRequest,
    Client,
    EvaluateRequest,
    EvaluationJobStatus,
    RateLimitError,
    RefineRequest,
    SkillPackage,
    SkillSession,
    TransportError,
    ValidationError,
)


def _skill_package_payload() -> dict[str, object]:
    return {
        "bundle_name": "sample-bundle",
        "files": [
            {"path": "SKILL.md", "content": "# Sample skill\n"},
            {"path": "scripts/run.sh", "content": "#!/usr/bin/env bash\necho ok\n"},
        ],
        "skills": [
            {
                "skill_id": "sample-skill",
                "skill_md": "# Sample skill\n",
            }
        ],
    }


def _build_response() -> dict[str, object]:
    return {
        "skill_package": _skill_package_payload(),
        "risk_flags": [],
        "activation_terms": ["sample"],
    }


def _refine_response() -> dict[str, object]:
    base_metrics = {
        "forced_pass_rate": 0.5,
        "autonomous_pass_rate": 0.5,
        "activation_rate": 0.5,
        "regression_count": 0,
        "token_estimate": 100,
    }
    delta_metrics = {
        "forced_pass_rate": 0.1,
        "autonomous_pass_rate": 0.1,
        "activation_rate": 0.1,
        "regression_count": 0,
        "token_estimate": 5,
    }
    return {
        "refined_skill_package": _skill_package_payload(),
        "accepted_edits": [],
        "rejected_edits": [],
        "complexity_delta": {
            "before_token_estimate": 100,
            "after_token_estimate": 105,
            "delta_tokens": 5,
            "before_editable_item_count": 2,
            "after_editable_item_count": 2,
            "delta_editable_items": 0,
        },
        "dev_revalidation": {
            "before_metrics": base_metrics,
            "after_metrics": base_metrics,
            "delta_metrics": delta_metrics,
        },
        "holdout_safety_result": {
            "before_metrics": base_metrics,
            "after_metrics": base_metrics,
            "delta_metrics": delta_metrics,
            "accepted": True,
            "within_tolerance": True,
        },
    }


def _make_sync_client(
    handler,
    *,
    api_key: str | None = "sk_live_test",
    bearer_token: str | None = None,
    model: str | None = None,
    model_provider_keys: dict[str, str] | None = None,
) -> Client:
    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.skillet.dev",
    )
    return Client(
        api_key=api_key,
        bearer_token=bearer_token,
        model=model,
        model_provider_keys=model_provider_keys,
        http_client=http_client,
    )


def _make_async_client(
    handler,
    *,
    api_key: str | None = "sk_live_test",
    bearer_token: str | None = None,
    model: str | None = None,
    model_provider_keys: dict[str, str] | None = None,
) -> AsyncClient:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.skillet.dev",
    )
    return AsyncClient(
        api_key=api_key,
        bearer_token=bearer_token,
        model=model,
        model_provider_keys=model_provider_keys,
        http_client=http_client,
    )


def _minimal_package() -> SkillPackage:
    return SkillPackage.from_api_payload(_skill_package_payload())


def test_public_surface_parity_for_sync_and_async_clients() -> None:
    expected_methods = {
        "build",
        "evaluate",
        "submit_evaluation",
        "get_evaluation_job",
        "wait_for_evaluation",
        "refine",
        "get_usage",
        "get_evaluations",
        "create_api_key",
        "list_api_keys",
        "rotate_api_key",
        "revoke_api_key",
    }

    for method_name in expected_methods:
        assert callable(getattr(Client, method_name, None))
        assert callable(getattr(AsyncClient, method_name, None))


def test_sdk_exports_session_and_not_workflow_alias() -> None:
    import skillet

    assert hasattr(skillet, "SkillSession")
    assert "SkillSession" in skillet.__all__


@pytest.mark.parametrize(
    ("status_code", "detail", "error_type"),
    [
        (401, "missing auth", AuthenticationError),
        (403, "forbidden", AuthorizationError),
        (402, "quota exceeded", RateLimitError),
        (429, "too many requests", RateLimitError),
        (422, [{"msg": "validation failed"}], ValidationError),
        (500, "server error", ApiError),
    ],
)
def test_sync_maps_http_status_codes_to_typed_errors(
    status_code: int,
    detail: object,
    error_type: type[Exception],
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": detail})

    client = _make_sync_client(handler)
    with pytest.raises(error_type):
        client.build("corpus")


def test_sync_maps_transport_failures_to_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection dropped", request=request)

    client = _make_sync_client(handler)
    with pytest.raises(TransportError):
        client.build("corpus")


def test_build_request_object_inherits_client_model_defaults() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_build_response())

    client = _make_sync_client(
        handler,
        model="gpt-5-nano",
        model_provider_keys={"openai": "sk-openai"},
    )

    client.build(BuildRequest(corpus_text="corpus"))
    assert captured["model"] == "gpt-5-nano"
    assert captured["model_provider_keys"] == {"openai": "sk-openai"}


def test_evaluate_and_refine_request_objects_inherit_client_defaults() -> None:
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured_bodies.append(body)
        if request.url.path == "/evaluate":
            return httpx.Response(200, json={"evaluation_report": {"metrics": {"pass_rate_delta": 0.2}}})
        if request.url.path == "/refine":
            return httpx.Response(200, json=_refine_response())
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    client = _make_sync_client(
        handler,
        model="gpt-5-nano",
        model_provider_keys={"openai": "sk-openai"},
    )

    package = _minimal_package()
    eval_request = EvaluateRequest(skill_package=package)
    refine_request = RefineRequest(
        skill_package=package,
        dev_tasks=[BenchmarkTask(task_id="dev-1", prompt="dev")],
        holdout_tasks=[BenchmarkTask(task_id="holdout-1", prompt="holdout")],
        edit_budget=1,
    )

    client.evaluate(eval_request)
    client.refine(refine_request)

    assert captured_bodies[0]["model"] == "gpt-5-nano"
    assert captured_bodies[0]["model_provider_keys"] == {"openai": "sk-openai"}
    assert captured_bodies[1]["model"] == "gpt-5-nano"
    assert captured_bodies[1]["model_provider_keys"] == {"openai": "sk-openai"}


def test_request_objects_reject_mixed_kwargs() -> None:
    package = _minimal_package()
    client = _make_sync_client(lambda _: httpx.Response(200, json=_build_response()))

    with pytest.raises(TypeError):
        client.build(BuildRequest(corpus_text="corpus"), bundle_target=1)

    with pytest.raises(TypeError):
        client.evaluate(EvaluateRequest(skill_package=package), bundle_ablation=False)

    with pytest.raises(TypeError):
        client.refine(
            RefineRequest(
                skill_package=package,
                dev_tasks=[BenchmarkTask(task_id="dev-1", prompt="dev")],
                holdout_tasks=[BenchmarkTask(task_id="holdout-1", prompt="holdout")],
                edit_budget=1,
            ),
            optimization_target="balanced",
        )


def test_sync_evaluate_polls_until_job_completion() -> None:
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.method == "POST" and request.url.path == "/evaluate":
            return httpx.Response(
                202,
                json={"job_id": "job-1", "status": EvaluationJobStatus.PENDING.value},
            )
        if request.method == "GET" and request.url.path == "/evaluate/job-1":
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(
                    200,
                    json={"job_id": "job-1", "status": EvaluationJobStatus.RUNNING.value},
                )
            return httpx.Response(
                200,
                json={
                    "job_id": "job-1",
                    "status": EvaluationJobStatus.COMPLETED.value,
                    "evaluation_report": {"metrics": {"pass_rate_delta": 0.5}},
                },
            )
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    client = _make_sync_client(handler)
    report = client.evaluate(_minimal_package(), poll_interval=0.0, timeout=5.0)

    assert report.metrics["pass_rate_delta"] == 0.5
    assert poll_count == 2


def test_sync_wait_for_evaluation_handles_timeout_and_failed_jobs() -> None:
    def timeout_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"job_id": "job-timeout", "status": EvaluationJobStatus.RUNNING.value},
        )

    timeout_client = _make_sync_client(timeout_handler)
    with pytest.raises(ApiError, match="Timed out"):
        timeout_client.wait_for_evaluation("job-timeout", poll_interval=0.0, timeout=0.0)

    def failed_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "job_id": "job-fail",
                "status": EvaluationJobStatus.FAILED.value,
                "error": "evaluation failed",
            },
        )

    failed_client = _make_sync_client(failed_handler)
    with pytest.raises(ApiError, match="evaluation failed"):
        failed_client.wait_for_evaluation("job-fail", poll_interval=0.0, timeout=5.0)


def test_async_client_maps_errors_and_polls_jobs() -> None:
    async def run() -> None:
        def auth_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "missing auth"})

        auth_client = _make_async_client(auth_handler)
        with pytest.raises(AuthenticationError):
            await auth_client.build("corpus")
        await auth_client.aclose()

        def transport_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection dropped", request=request)

        transport_client = _make_async_client(transport_handler)
        with pytest.raises(TransportError):
            await transport_client.build("corpus")
        await transport_client.aclose()

        poll_count = 0

        def polling_handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            if request.method == "POST" and request.url.path == "/evaluate":
                return httpx.Response(
                    202,
                    json={"job_id": "job-async", "status": EvaluationJobStatus.PENDING.value},
                )
            if request.method == "GET" and request.url.path == "/evaluate/job-async":
                poll_count += 1
                if poll_count == 1:
                    return httpx.Response(
                        200,
                        json={"job_id": "job-async", "status": EvaluationJobStatus.RUNNING.value},
                    )
                return httpx.Response(
                    200,
                    json={
                        "job_id": "job-async",
                        "status": EvaluationJobStatus.COMPLETED.value,
                        "evaluation_report": {"metrics": {"pass_rate_delta": 0.9}},
                    },
                )
            raise AssertionError(f"Unexpected request path: {request.url.path}")

        polling_client = _make_async_client(polling_handler)
        report = await polling_client.evaluate(_minimal_package(), poll_interval=0.0, timeout=5.0)
        await polling_client.aclose()

        assert report.metrics["pass_rate_delta"] == 0.9
        assert poll_count == 2

    asyncio.run(run())


def test_bundle_export_extract_to_save_to_and_write_json(tmp_path: Path) -> None:
    package = _minimal_package()
    bundle = package.bundle()

    export_dir = bundle.extract_to(tmp_path / "bundle")
    assert (export_dir / "SKILL.md").read_text(encoding="utf-8") == "# Sample skill\n"
    assert (export_dir / "scripts" / "run.sh").exists()

    archive_path = bundle.save_to(tmp_path / "bundle.zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == ["SKILL.md", "scripts/run.sh"]

    json_path = package.write_json(tmp_path / "skill-package.json")
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["bundle_name"] == "sample-bundle"


def test_bundle_extract_to_rejects_file_target(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        _minimal_package().bundle().extract_to(target)


def test_skill_session_build_evaluate_refine_flow() -> None:
    class FakeClient:
        def build(self, _: object) -> SkillPackage:
            return _minimal_package()

        def evaluate(self, _: SkillPackage, **__: object) -> object:
            from skillet import EvaluationReport

            return EvaluationReport.model_validate({"metrics": {"pass_rate_delta": 0.1}})

        def refine(self, _: SkillPackage, **__: object) -> object:
            from skillet import RefineResult

            return RefineResult.from_api_payload(_refine_response())

    session = SkillSession(FakeClient()).from_corpus("corpus")
    built = session.build()
    report = session.evaluate()
    refined = session.refine(
        dev_tasks=[BenchmarkTask(task_id="dev-1", prompt="dev")],
        holdout_tasks=[BenchmarkTask(task_id="holdout-1", prompt="holdout")],
        edit_budget=1,
    )

    assert built.bundle_name == "sample-bundle"
    assert report.metrics["pass_rate_delta"] == 0.1
    assert refined.holdout_safety_result.accepted is True
