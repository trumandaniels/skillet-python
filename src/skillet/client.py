from __future__ import annotations

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
    **kwargs: Any,
) -> BuildRequest:
    if isinstance(corpus_text, BuildRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        return corpus_text
    return BuildRequest(corpus_text=corpus_text, **kwargs)


def _coerce_skill_package(
    skill_package: SkillPackage | dict[str, Any],
) -> SkillPackage:
    if isinstance(skill_package, SkillPackage):
        return skill_package
    return SkillPackage.from_api_payload(skill_package)


def _coerce_evaluate_request(
    skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
    **kwargs: Any,
) -> EvaluateRequest:
    if isinstance(skill_package, EvaluateRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        return skill_package
    return EvaluateRequest(skill_package=_coerce_skill_package(skill_package), **kwargs)


def _coerce_refine_request(
    skill_package: SkillPackage | RefineRequest | dict[str, Any],
    **kwargs: Any,
) -> RefineRequest:
    if isinstance(skill_package, RefineRequest):
        if kwargs:
            raise TypeError("Keyword arguments are not supported when passing a request object")
        return skill_package
    return RefineRequest(skill_package=_coerce_skill_package(skill_package), **kwargs)


class Client:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        base_url: str = "https://api.skillet.dev",
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "skillet-python-sdk/0.1"},
        )

    def close(self) -> None:
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
        request = _coerce_build_request(corpus_text, **kwargs)
        response = self._request_json(
            "POST",
            "/build",
            headers=self._pipeline_headers(),
            json_body=request.model_dump(mode="json"),
        )
        return SkillPackage.from_build_response(response)

    def evaluate(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        **kwargs: Any,
    ) -> EvaluationReport:
        request = _coerce_evaluate_request(skill_package, **kwargs)
        response = self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return EvaluationReport.model_validate(response["evaluation_report"])

    def refine(
        self,
        skill_package: SkillPackage | RefineRequest | dict[str, Any],
        **kwargs: Any,
    ) -> RefineResult:
        request = _coerce_refine_request(skill_package, **kwargs)
        response = self._request_json(
            "POST",
            "/refine",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return RefineResult.from_api_payload(response)

    def get_usage(self, month: str | None = None) -> UsageSummary:
        response = self._request_json(
            "GET",
            "/app/usage",
            headers=self._app_headers(),
            params={"month": month} if month else None,
        )
        return UsageSummary.model_validate(response)

    def create_api_key(self, label: str) -> IssuedApiKey:
        response = self._request_json(
            "POST",
            "/app/api-keys",
            headers=self._app_headers(),
            json_body={"label": label},
        )
        return IssuedApiKey.model_validate(response)

    def list_api_keys(self) -> list[ApiKeySummary]:
        response = self._request_json(
            "GET",
            "/app/api-keys",
            headers=self._app_headers(),
        )
        return [ApiKeySummary.model_validate(item) for item in response]

    def rotate_api_key(self, key_id: str) -> IssuedApiKey:
        response = self._request_json(
            "POST",
            f"/app/api-keys/{key_id}/rotate",
            headers=self._app_headers(),
        )
        return IssuedApiKey.model_validate(response)

    def revoke_api_key(self, key_id: str) -> ApiKeySummary:
        response = self._request_json(
            "DELETE",
            f"/app/api-keys/{key_id}",
            headers=self._app_headers(),
        )
        return ApiKeySummary.model_validate(response)


class AsyncClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        base_url: str = "https://api.skillet.dev",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "skillet-python-sdk/0.1"},
        )

    async def aclose(self) -> None:
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
        request = _coerce_build_request(corpus_text, **kwargs)
        response = await self._request_json(
            "POST",
            "/build",
            headers=self._pipeline_headers(),
            json_body=request.model_dump(mode="json"),
        )
        return SkillPackage.from_build_response(response)

    async def evaluate(
        self,
        skill_package: SkillPackage | EvaluateRequest | dict[str, Any],
        **kwargs: Any,
    ) -> EvaluationReport:
        request = _coerce_evaluate_request(skill_package, **kwargs)
        response = await self._request_json(
            "POST",
            "/evaluate",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return EvaluationReport.model_validate(response["evaluation_report"])

    async def refine(
        self,
        skill_package: SkillPackage | RefineRequest | dict[str, Any],
        **kwargs: Any,
    ) -> RefineResult:
        request = _coerce_refine_request(skill_package, **kwargs)
        response = await self._request_json(
            "POST",
            "/refine",
            headers=self._pipeline_headers(),
            json_body=request.to_api_payload(),
        )
        return RefineResult.from_api_payload(response)

    async def get_usage(self, month: str | None = None) -> UsageSummary:
        response = await self._request_json(
            "GET",
            "/app/usage",
            headers=self._app_headers(),
            params={"month": month} if month else None,
        )
        return UsageSummary.model_validate(response)

    async def create_api_key(self, label: str) -> IssuedApiKey:
        response = await self._request_json(
            "POST",
            "/app/api-keys",
            headers=self._app_headers(),
            json_body={"label": label},
        )
        return IssuedApiKey.model_validate(response)

    async def list_api_keys(self) -> list[ApiKeySummary]:
        response = await self._request_json(
            "GET",
            "/app/api-keys",
            headers=self._app_headers(),
        )
        return [ApiKeySummary.model_validate(item) for item in response]

    async def rotate_api_key(self, key_id: str) -> IssuedApiKey:
        response = await self._request_json(
            "POST",
            f"/app/api-keys/{key_id}/rotate",
            headers=self._app_headers(),
        )
        return IssuedApiKey.model_validate(response)

    async def revoke_api_key(self, key_id: str) -> ApiKeySummary:
        response = await self._request_json(
            "DELETE",
            f"/app/api-keys/{key_id}",
            headers=self._app_headers(),
        )
        return ApiKeySummary.model_validate(response)
