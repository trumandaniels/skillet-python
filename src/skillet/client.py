from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .exceptions import (
    ApiError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    TransportError,
    ValidationError,
)
from .models import (
    ApiKeySummary,
    BuildRequest,
    EvaluateRequest,
    EvaluationHistory,
    EvaluationJob,
    EvaluationJobSubmission,
    EvaluationReport,
    IssuedApiKey,
    RefineResult,
    RefineRequest,
    SkillPackage,
    UsageSummary,
)


def _detail_from_response(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        return response.text or None
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return payload


def _message_from_detail(detail: Any, fallback: str) -> str:
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and "msg" in first:
            return str(first["msg"])
    return fallback


def _raise_for_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    detail = _detail_from_response(response)
    message = _message_from_detail(
        detail,
        f"Skillet API request failed with status {response.status_code}",
    )
    if response.status_code == 401:
        raise AuthenticationError(message, status_code=response.status_code, detail=detail)
    if response.status_code == 403:
        raise AuthorizationError(message, status_code=response.status_code, detail=detail)
    if response.status_code in {402, 429}:
        raise RateLimitError(message, status_code=response.status_code, detail=detail)
    if response.status_code == 422:
        raise ValidationError(message, status_code=response.status_code, detail=detail)
    raise ApiError(message, status_code=response.status_code, detail=detail)


def _coerce_build_request(
    corpus_text: str | BuildRequest,
    *,
    client_model_provider_keys: dict[str, str] | None = None,
    client_model: str | None = None,
    **kwargs: Any,
) -> BuildRequest:
    if isinstance(corpus_text, BuildRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        updates: dict[str, Any] = {}
        if corpus_text.model_provider_keys is None and client_model_provider_keys is not None:
            updates["model_provider_keys"] = client_model_provider_keys
        if corpus_text.model is None and client_model is not None:
            updates["model"] = client_model
        if updates:
            corpus_text = corpus_text.model_copy(update=updates)
        return corpus_text
    if "model_provider_keys" not in kwargs and client_model_provider_keys is not None:
        kwargs["model_provider_keys"] = client_model_provider_keys
    if "model" not in kwargs and client_model is not None:
        kwargs["model"] = client_model
    return BuildRequest(corpus_text=corpus_text, **kwargs)


def _coerce_skill_package(
    skill_package: SkillPackage | dict[str, Any],
) -> SkillPackage:
    if isinstance(skill_package, SkillPackage):
        return skill_package
    return SkillPackage.from_api_payload(skill_package)


def _coerce_evaluate_request(
    skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
    *,
    client_model_provider_keys: dict[str, str] | None = None,
    client_model: str | None = None,
    **kwargs: Any,
) -> EvaluateRequest:
    if isinstance(skill_package, EvaluateRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        updates: dict[str, Any] = {}
        if skill_package.model_provider_keys is None and client_model_provider_keys is not None:
            updates["model_provider_keys"] = client_model_provider_keys
        if skill_package.model is None and client_model is not None:
            updates["model"] = client_model
        if updates:
            skill_package = skill_package.model_copy(update=updates)
        return skill_package
    if "model_provider_keys" not in kwargs and client_model_provider_keys is not None:
        kwargs["model_provider_keys"] = client_model_provider_keys
    if "model" not in kwargs and client_model is not None:
        kwargs["model"] = client_model
    return EvaluateRequest(skill_package=_coerce_skill_package(skill_package), **kwargs)


def _coerce_refine_request(
    skill_package: SkillPackage | RefineRequest | dict[str, Any],
    *,
    client_model_provider_keys: dict[str, str] | None = None,
    client_model: str | None = None,
    **kwargs: Any,
) -> RefineRequest:
    if isinstance(skill_package, RefineRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        updates: dict[str, Any] = {}
        if skill_package.model_provider_keys is None and client_model_provider_keys is not None:
            updates["model_provider_keys"] = client_model_provider_keys
        if skill_package.model is None and client_model is not None:
            updates["model"] = client_model
        if updates:
            skill_package = skill_package.model_copy(update=updates)
        return skill_package
    if "model_provider_keys" not in kwargs and client_model_provider_keys is not None:
        kwargs["model_provider_keys"] = client_model_provider_keys
    if "model" not in kwargs and client_model is not None:
        kwargs["model"] = client_model
    return RefineRequest(skill_package=_coerce_skill_package(skill_package), **kwargs)


class Client:
    """Synchronous typed HTTP client for the Skillet API.

    Construct a client with either an ``api_key`` (for pipeline calls) or a
    ``bearer_token`` (for ``/app/*`` management routes), or both when you need
    a single client that can do everything.

    The client owns its ``httpx.Client`` by default and closes it when
    ``close()`` is called or the context manager exits.  Pass a custom
    ``http_client`` to share a session or inject a test transport.

    Args:
        api_key: Tenant-owned API key sent as ``X-API-Key``.  Required for
            ``build``, ``evaluate``, and ``refine`` unless ``bearer_token`` is provided.
        bearer_token: JWT bearer token sent as ``Authorization: Bearer``.
            Required for ``/app/*`` routes; also accepted by pipeline routes.
        base_url: API base URL.  Defaults to ``https://api.skillet.dev``.
        timeout: ``httpx`` request timeout in seconds.  Defaults to ``30.0``.
        http_client: Optional injected ``httpx.Client`` for tests or advanced
            transport control.  When provided, the caller is responsible for
            closing it.
        model_provider_keys: Optional per-client provider API keys passed
            through on every pipeline call unless overridden per request.
        model: Optional default model passed through on every pipeline call
            unless overridden per request.

    Example:
        ```python
        from skillet import Client

        client = Client(
            api_key="sk_live_...",
            bearer_token="your-bearer-token",
        )

        # Use as a context manager for automatic cleanup
        with Client(api_key="sk_live_...") as client:
            package = client.build("Your corpus text here...")
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        base_url: str = "https://api.skillet.dev",
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        model_provider_keys: dict[str, str] | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.model_provider_keys = model_provider_keys
        self.model = model
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "skillet-python-sdk/0.1"},
        )

    def close(self) -> None:
        """Close the underlying ``httpx.Client`` when this instance owns it.

        If you injected a custom ``http_client``, this method is a no-op and the
        caller remains responsible for cleaning it up.

        Example:
            ```python
            client = Client(api_key="sk_live_...")
            try:
                package = client.build("Your corpus text here...")
            finally:
                client.close()
            ```
        """
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _pipeline_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        raise AuthenticationError("An API key or bearer token is required for pipeline endpoints")

    def _app_headers(self) -> dict[str, str]:
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        raise AuthenticationError("A bearer token is required for /app endpoints")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                path,
                headers=headers,
                json=json_body,
                params=params,
            )
        except httpx.RequestError as exc:
            raise TransportError(str(exc), detail=exc) from exc
        _raise_for_response(response)
        return response.json()

    def build(
        self,
        corpus_text: str | BuildRequest,
        **kwargs: Any,
    ) -> SkillPackage:
        """Build a ``SkillPackage`` from raw corpus text.

        Use this when you already have corpus text as a single string and want
        direct control over build options without using ``SkillSession``.

        Args:
            corpus_text: Raw domain corpus as a string, or a pre-validated
                ``BuildRequest`` object.  When passing a ``BuildRequest``,
                keyword arguments are not accepted.
            **kwargs: Keyword arguments forwarded to ``BuildRequest``.  Accepted
                keys: ``overlap_ratio``, ``target_runtime``, ``bundle_target``,
                ``length_profile``, ``emit_scripts``, ``emit_checks``,
                ``model_provider_keys``, ``model``.

        Returns:
            A ``SkillPackage`` containing the compiled skills, bundle manifest,
            complexity report, risk flags, and recommended runtime profile.

        Example:
            ```python
            package = client.build(
                "Instrumental variables help identify causal effects under endogeneity.",
                bundle_target=2,
                length_profile="moderate",
            )
            ```
        """
        request = _coerce_build_request(
            corpus_text,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = self._request_json(
            "POST",
            "/build",
            headers=self._pipeline_headers(),
            json_body=request.model_dump(mode="json", exclude_none=True),
        )
        return SkillPackage.from_build_response(response)

    def evaluate(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Evaluate a previously built package against the benchmark harness.

        Use this when you already have a ``SkillPackage`` and want explicit control
        over activation policies and ablations.

        Args:
            skill_package: A ``SkillPackage``, ``EvaluateRequest``, or a raw
                ``dict`` containing a package payload.  When passing an
                ``EvaluateRequest``, keyword arguments are not accepted.
            **kwargs: Keyword arguments forwarded to ``EvaluateRequest``.  Accepted
                keys: ``activation_policies``, ``bundle_ablation``,
                ``length_ablation``, ``discovery_ablation``, ``suite``,
                ``run_profile``, ``compare_to``, ``baseline_eval_id``,
                ``quality_gate``, ``max_tokens``, ``max_tool_calls``,
                ``max_wall_time_seconds``, ``max_estimated_cost_usd``,
                ``model_provider_keys``, ``model``.

        Returns:
            An ``EvaluationReport`` with pass-rate delta, regression count,
            cost metrics, and ablation results.

        Example:
            ```python
            report = client.evaluate(
                package,
                activation_policies=["forced", "autonomous"],
                bundle_ablation=True,
                length_ablation=True,
            )
            ```
        """
        request = _coerce_evaluate_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        if isinstance(response, dict) and "evaluation_report" in response:
            return EvaluationReport.model_validate(response["evaluation_report"])
        submission = EvaluationJobSubmission.model_validate(response)
        return self.wait_for_evaluation(
            submission.job_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def submit_evaluation(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        **kwargs: Any,
    ) -> EvaluationJobSubmission:
        """Submit an evaluation job without waiting for completion.

        Use this when you want explicit control over polling, background job
        tracking, or external orchestration.

        Args:
            skill_package: A ``SkillPackage``, ``EvaluateRequest``, or raw
                ``dict`` payload to evaluate.
            **kwargs: Keyword arguments forwarded to ``EvaluateRequest``.

        Returns:
            An ``EvaluationJobSubmission`` containing the queued ``job_id``.

        Example:
            ```python
            submission = client.submit_evaluation(
                package,
                activation_policies=["forced", "autonomous"],
            )
            print(submission.job_id)
            ```
        """
        request = _coerce_evaluate_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return EvaluationJobSubmission.model_validate(response)

    def get_evaluation_job(self, job_id: str) -> EvaluationJob:
        """Fetch the current state of an evaluation job.

        Args:
            job_id: Identifier returned by ``submit_evaluation()``.

        Returns:
            An ``EvaluationJob`` with status, error details, and report payload
            when the job has completed.

        Example:
            ```python
            job = client.get_evaluation_job("evaljob_123")
            print(job.status)
            ```
        """
        response = self._request_json(
            "GET",
            f"/evaluate/{job_id}",
            headers=self._pipeline_headers(),
        )
        return EvaluationJob.model_validate(response)

    def wait_for_evaluation(
        self,
        job_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> EvaluationReport:
        """Poll an evaluation job until it completes or fails.

        Args:
            job_id: Identifier returned by ``submit_evaluation()``.
            poll_interval: Seconds to wait between status checks.
            timeout: Optional maximum total wait time in seconds.

        Returns:
            The completed ``EvaluationReport``.

        Example:
            ```python
            submission = client.submit_evaluation(package)
            report = client.wait_for_evaluation(submission.job_id, poll_interval=2.0)
            print(report.metrics["pass_rate_delta"])
            ```
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise ApiError(f"Timed out while waiting for evaluation job {job_id}")
            job = self.get_evaluation_job(job_id)
            if job.status == "completed":
                if job.evaluation_report is None:
                    raise ApiError(f"Evaluation job {job_id} completed without a report")
                return job.evaluation_report
            if job.status == "failed":
                raise ApiError(job.error or f"Evaluation job {job_id} failed")
            time.sleep(max(poll_interval, 0.0))

    def refine(
        self,
        skill_package: SkillPackage | RefineRequest | dict[str, Any],
        **kwargs: Any,
    ) -> RefineResult:
        """Apply constrained edits to a package and revalidate against dev and holdout tasks.

        Use this when evaluation results suggest a change in wording, checks, or
        decision rules.  The server only accepts edits that improve the chosen
        optimization target without regressing holdout safety.

        Args:
            skill_package: A ``SkillPackage``, ``RefineRequest``, or raw ``dict``.
                When passing a ``RefineRequest``, keyword arguments are not accepted.
            **kwargs: Keyword arguments forwarded to ``RefineRequest``.  Required
                keys: ``proposed_edits``, ``dev_tasks``, ``holdout_tasks``,
                ``edit_budget``.  Optional: ``optimization_target``,
                ``model_provider_keys``, ``model``.

        Returns:
            A ``RefineResult`` containing the refined package, accepted/rejected
            edits, complexity delta, and holdout safety result.

        Example:
            ```python
            refined = client.refine(
                package,
                proposed_edits=[
                    {"section": "decision_rules", "value": ["Abort weak IV runs when F < 10"]},
                ],
                dev_tasks=[{"task_id": "dev-1", "prompt": "Stop weak IV runs"}],
                holdout_tasks=[{"task_id": "holdout-1", "prompt": "Stop weak IV runs"}],
                edit_budget=1,
            )
            ```
        """
        request = _coerce_refine_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = self._request_json(
            "POST",
            "/refine",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return RefineResult.from_api_payload(response)

    def get_usage(self, month: str | None = None) -> UsageSummary:
        """Fetch monthly org-scoped usage and remaining quota.

        Args:
            month: Optional billing month in ``YYYY-MM`` format.  Defaults to
                the current calendar month.

        Returns:
            A ``UsageSummary`` with usage, limits, remaining runs, and
            per-key attribution.

        Example:
            ```python
            usage = client.get_usage(month="2026-03")
            print(usage.remaining)
            ```
        """
        response = self._request_json(
            "GET",
            "/app/usage",
            headers=self._app_headers(),
            params={"month": month} if month else None,
        )
        return UsageSummary.model_validate(response)

    def get_evaluations(self, limit: int = 50) -> EvaluationHistory:
        """Fetch recent evaluation history for the authenticated organization.

        Args:
            limit: Maximum number of most-recent evaluation runs to return.

        Returns:
            An ``EvaluationHistory`` with recent runs, gate outcomes,
            baseline metadata, and runtime summaries.

        Example:
            ```python
            history = client.get_evaluations(limit=25)
            print(history.evaluations[0].gate_result.status)
            ```
        """
        response = self._request_json(
            "GET",
            "/app/evaluations",
            headers=self._app_headers(),
            params={"limit": limit},
        )
        return EvaluationHistory.model_validate(response)

    def create_api_key(self, label: str) -> IssuedApiKey:
        """Create a new tenant-owned API key.

        Use this to bootstrap the first pipeline key from a bearer-authenticated
        client, or to create separate keys for different apps or environments.
        The raw secret is returned once only — store it immediately.

        Args:
            label: Human-readable key label.

        Returns:
            An ``IssuedApiKey`` containing the key summary and the one-time ``secret``.

        Example:
            ```python
            issued = client.create_api_key("production-worker")
            print(issued.secret)  # store this — it won't be shown again
            ```
        """
        response = self._request_json(
            "POST",
            "/app/api-keys",
            headers=self._app_headers(),
            json_body={"label": label},
        )
        return IssuedApiKey.model_validate(response)

    def list_api_keys(self) -> list[ApiKeySummary]:
        """List all API keys visible to the authenticated tenant.

        Returns:
            A list of ``ApiKeySummary`` objects with current-month usage metrics.

        Example:
            ```python
            keys = client.list_api_keys()
            ```
        """
        response = self._request_json(
            "GET",
            "/app/api-keys",
            headers=self._app_headers(),
        )
        return [ApiKeySummary.model_validate(item) for item in response]

    def rotate_api_key(self, key_id: str) -> IssuedApiKey:
        """Rotate an existing API key and return the replacement secret.

        The old key is revoked immediately.  The new secret is returned once only.

        Args:
            key_id: Server-assigned key identifier.

        Returns:
            An ``IssuedApiKey`` with the new one-time ``secret``.

        Example:
            ```python
            rotated = client.rotate_api_key("key_01abc")
            print(rotated.secret)
            ```
        """
        response = self._request_json(
            "POST",
            f"/app/api-keys/{key_id}/rotate",
            headers=self._app_headers(),
        )
        return IssuedApiKey.model_validate(response)

    def revoke_api_key(self, key_id: str) -> ApiKeySummary:
        """Permanently revoke an API key.

        Args:
            key_id: Server-assigned key identifier.

        Returns:
            The ``ApiKeySummary`` with ``revoked_at`` populated.

        Example:
            ```python
            revoked = client.revoke_api_key("key_01abc")
            print(revoked.revoked_at)
            ```
        """
        response = self._request_json(
            "DELETE",
            f"/app/api-keys/{key_id}",
            headers=self._app_headers(),
        )
        return ApiKeySummary.model_validate(response)


class AsyncClient:
    """Async equivalent of ``Client`` for use with ``asyncio``.

    Mirrors the ``Client`` method surface one-for-one.  Use it when your
    application already uses ``asyncio`` and you want non-blocking HTTP calls.

    Args:
        api_key: Tenant-owned API key sent as ``X-API-Key``.
        bearer_token: JWT bearer token sent as ``Authorization: Bearer``.
        base_url: API base URL.  Defaults to ``https://api.skillet.dev``.
        timeout: ``httpx`` request timeout in seconds.  Defaults to ``30.0``.
        http_client: Optional injected ``httpx.AsyncClient``.
        model_provider_keys: Optional per-client provider API keys passed
            through on every pipeline call unless overridden per request.
        model: Optional default model passed through on every pipeline call
            unless overridden per request.

    Example:
        ```python
        from skillet import AsyncClient

        async with AsyncClient(api_key="sk_live_...") as client:
            package = await client.build("Your corpus text here...")
            report = await client.evaluate(package)
            usage = await client.get_usage()
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        base_url: str = "https://api.skillet.dev",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        model_provider_keys: dict[str, str] | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.model_provider_keys = model_provider_keys
        self.model = model
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "skillet-python-sdk/0.1"},
        )

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` when this instance owns it.

        If you injected a custom ``http_client``, this method is a no-op and the
        caller remains responsible for cleaning it up.

        Example:
            ```python
            client = AsyncClient(api_key="sk_live_...")
            try:
                package = await client.build("Your corpus text here...")
            finally:
                await client.aclose()
            ```
        """
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def _pipeline_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"X-API-Key": self.api_key}
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        raise AuthenticationError("An API key or bearer token is required for pipeline endpoints")

    def _app_headers(self) -> dict[str, str]:
        if self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        raise AuthenticationError("A bearer token is required for /app endpoints")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                path,
                headers=headers,
                json=json_body,
                params=params,
            )
        except httpx.RequestError as exc:
            raise TransportError(str(exc), detail=exc) from exc
        _raise_for_response(response)
        return response.json()

    async def build(
        self,
        corpus_text: str | BuildRequest,
        **kwargs: Any,
    ) -> SkillPackage:
        """Async version of ``Client.build``.

        Use the same inputs as ``Client.build`` but ``await`` the network call.

        Example:
            ```python
            package = await client.build(
                "Instrumental variables help identify causal effects under endogeneity.",
                bundle_target=2,
                length_profile="moderate",
            )
            ```
        """
        request = _coerce_build_request(
            corpus_text,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = await self._request_json(
            "POST",
            "/build",
            headers=self._pipeline_headers(),
            json_body=request.model_dump(mode="json", exclude_none=True),
        )
        return SkillPackage.from_build_response(response)

    async def evaluate(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Async version of ``Client.evaluate``.

        Use the same inputs as ``Client.evaluate`` but ``await`` the network call.

        Example:
            ```python
            report = await client.evaluate(
                package,
                activation_policies=["forced", "autonomous"],
                bundle_ablation=True,
                length_ablation=True,
            )
            ```
        """
        request = _coerce_evaluate_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = await self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        if isinstance(response, dict) and "evaluation_report" in response:
            return EvaluationReport.model_validate(response["evaluation_report"])
        submission = EvaluationJobSubmission.model_validate(response)
        return await self.wait_for_evaluation(
            submission.job_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def submit_evaluation(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        **kwargs: Any,
    ) -> EvaluationJobSubmission:
        """Async version of ``Client.submit_evaluation``.

        Submit an evaluation job and return immediately with the queued
        ``job_id``.

        Example:
            ```python
            submission = await client.submit_evaluation(
                package,
                activation_policies=["forced", "autonomous"],
            )
            print(submission.job_id)
            ```
        """
        request = _coerce_evaluate_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = await self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return EvaluationJobSubmission.model_validate(response)

    async def get_evaluation_job(self, job_id: str) -> EvaluationJob:
        """Async version of ``Client.get_evaluation_job``.

        Example:
            ```python
            job = await client.get_evaluation_job("evaljob_123")
            print(job.status)
            ```
        """
        response = await self._request_json(
            "GET",
            f"/evaluate/{job_id}",
            headers=self._pipeline_headers(),
        )
        return EvaluationJob.model_validate(response)

    async def wait_for_evaluation(
        self,
        job_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> EvaluationReport:
        """Async version of ``Client.wait_for_evaluation``.

        Example:
            ```python
            submission = await client.submit_evaluation(package)
            report = await client.wait_for_evaluation(submission.job_id, poll_interval=2.0)
            print(report.metrics["pass_rate_delta"])
            ```
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise ApiError(f"Timed out while waiting for evaluation job {job_id}")
            job = await self.get_evaluation_job(job_id)
            if job.status == "completed":
                if job.evaluation_report is None:
                    raise ApiError(f"Evaluation job {job_id} completed without a report")
                return job.evaluation_report
            if job.status == "failed":
                raise ApiError(job.error or f"Evaluation job {job_id} failed")
            await asyncio.sleep(max(poll_interval, 0.0))

    async def refine(
        self,
        skill_package: SkillPackage | RefineRequest | dict[str, Any],
        **kwargs: Any,
    ) -> RefineResult:
        """Async version of ``Client.refine``.

        Use the same inputs as ``Client.refine`` but ``await`` the network call.

        Example:
            ```python
            refined = await client.refine(
                package,
                proposed_edits=[
                    {"section": "decision_rules", "value": ["Abort weak IV runs when F < 10"]},
                ],
                dev_tasks=[{"task_id": "dev-1", "prompt": "Stop weak IV runs"}],
                holdout_tasks=[{"task_id": "holdout-1", "prompt": "Stop weak IV runs"}],
                edit_budget=1,
            )
            ```
        """
        request = _coerce_refine_request(
            skill_package,
            client_model_provider_keys=self.model_provider_keys,
            client_model=self.model,
            **kwargs,
        )
        response = await self._request_json(
            "POST",
            "/refine",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return RefineResult.from_api_payload(response)

    async def get_usage(self, month: str | None = None) -> UsageSummary:
        """Async version of ``Client.get_usage``.

        Example:
            ```python
            usage = await client.get_usage(month="2026-03")
            print(usage.remaining)
            ```
        """
        response = await self._request_json(
            "GET",
            "/app/usage",
            headers=self._app_headers(),
            params={"month": month} if month else None,
        )
        return UsageSummary.model_validate(response)

    async def get_evaluations(self, limit: int = 50) -> EvaluationHistory:
        """Async version of ``Client.get_evaluations``.

        Example:
            ```python
            history = await client.get_evaluations(limit=25)
            print(history.evaluations[0].baseline.status)
            ```
        """
        response = await self._request_json(
            "GET",
            "/app/evaluations",
            headers=self._app_headers(),
            params={"limit": limit},
        )
        return EvaluationHistory.model_validate(response)

    async def create_api_key(self, label: str) -> IssuedApiKey:
        """Async version of ``Client.create_api_key``.

        Example:
            ```python
            issued = await client.create_api_key("production-worker")
            print(issued.secret)
            ```
        """
        response = await self._request_json(
            "POST",
            "/app/api-keys",
            headers=self._app_headers(),
            json_body={"label": label},
        )
        return IssuedApiKey.model_validate(response)

    async def list_api_keys(self) -> list[ApiKeySummary]:
        """Async version of ``Client.list_api_keys``.

        Example:
            ```python
            keys = await client.list_api_keys()
            ```
        """
        response = await self._request_json(
            "GET",
            "/app/api-keys",
            headers=self._app_headers(),
        )
        return [ApiKeySummary.model_validate(item) for item in response]

    async def rotate_api_key(self, key_id: str) -> IssuedApiKey:
        """Async version of ``Client.rotate_api_key``.

        Example:
            ```python
            rotated = await client.rotate_api_key("key_01abc")
            print(rotated.secret)
            ```
        """
        response = await self._request_json(
            "POST",
            f"/app/api-keys/{key_id}/rotate",
            headers=self._app_headers(),
        )
        return IssuedApiKey.model_validate(response)

    async def revoke_api_key(self, key_id: str) -> ApiKeySummary:
        """Async version of ``Client.revoke_api_key``.

        Example:
            ```python
            revoked = await client.revoke_api_key("key_01abc")
            print(revoked.revoked_at)
            ```
        """
        response = await self._request_json(
            "DELETE",
            f"/app/api-keys/{key_id}",
            headers=self._app_headers(),
        )
        return ApiKeySummary.model_validate(response)
