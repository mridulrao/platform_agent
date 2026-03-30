"""
ServiceNow tools for LiveKit agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
from dotenv import load_dotenv
from livekit.agents import function_tool


load_dotenv()

logger = logging.getLogger(__name__)


def _success(data: Any, status_code: int = 200, message: str = "OK") -> dict[str, Any]:
    return {
        "ok": True,
        "status_code": status_code,
        "message": message,
        "data": data,
    }


def _error(
    message: str,
    *,
    status_code: int = 500,
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "status_code": status_code,
        "message": message,
    }
    if details is not None:
        payload["details"] = details
    return payload


@dataclass(slots=True)
class ServiceNowConfig:
    instance: str
    username: str
    password: str
    incident_table: str = "incident"
    user_table: str = "sys_user"
    request_table: str = "sc_request"
    base_path: str = "/api/now/table"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 0.5

    @property
    def base_url(self) -> str:
        return f"https://{self.instance}.service-now.com{self.base_path}"

    @classmethod
    def from_env(cls) -> "ServiceNowConfig":
        instance = os.getenv("SNOW_INSTANCE", "").strip()
        username = os.getenv("SNOW_USERNAME", "").strip()
        password = os.getenv("SNOW_PASSWORD", "").strip()
        if not instance or not username or not password:
            missing = [
                name
                for name, value in (
                    ("SNOW_INSTANCE", instance),
                    ("SNOW_USERNAME", username),
                    ("SNOW_PASSWORD", password),
                )
                if not value
            ]
            raise ValueError(
                f"Missing ServiceNow configuration: {', '.join(missing)}"
            )

        return cls(
            instance=instance,
            username=username,
            password=password,
            incident_table=os.getenv("SNOW_INCIDENT_TABLE", "incident").strip() or "incident",
            user_table=os.getenv("SNOW_USER_TABLE", "sys_user").strip() or "sys_user",
            request_table=os.getenv("SNOW_REQUEST_TABLE", "sc_request").strip() or "sc_request",
            base_path=os.getenv("SNOW_BASE_PATH", "/api/now/table").strip() or "/api/now/table",
            timeout_seconds=int(os.getenv("SNOW_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("SNOW_MAX_RETRIES", "3")),
            retry_delay_seconds=float(os.getenv("SNOW_RETRY_DELAY_SECONDS", "0.5")),
        )


class ServiceNowClient:
    def __init__(self, config: ServiceNowConfig) -> None:
        self.config = config

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        expected_statuses: tuple[int, ...] = (200, 201),
    ) -> dict[str, Any]:
        auth = aiohttp.BasicAuth(self.config.username, self.config.password)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        last_error: str | None = None
        retryable_statuses = {408, 429, 500, 502, 503, 504}

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    auth=auth,
                    timeout=timeout,
                    headers=headers,
                ) as session:
                    async with session.request(
                        method=method,
                        url=f"{self.config.base_url}/{path.lstrip('/')}",
                        params=params,
                        json=json_body,
                        ssl=False,
                    ) as response:
                        text = await response.text()

                        if response.status in expected_statuses:
                            if not text.strip():
                                return _success({}, status_code=response.status)

                            try:
                                payload = await response.json()
                            except aiohttp.ContentTypeError:
                                return _error(
                                    "ServiceNow returned a non-JSON response",
                                    status_code=response.status,
                                    details=text,
                                )

                            return _success(
                                payload.get("result", payload),
                                status_code=response.status,
                            )

                        if response.status in retryable_statuses and attempt < self.config.max_retries:
                            await asyncio.sleep(
                                self.config.retry_delay_seconds * (2 ** (attempt - 1))
                            )
                            continue

                        details: Any
                        try:
                            details = await response.json()
                        except aiohttp.ContentTypeError:
                            details = text

                        return _error(
                            "ServiceNow request failed",
                            status_code=response.status,
                            details=details,
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)
                logger.warning(
                    "ServiceNow request attempt %s/%s failed: %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_delay_seconds * (2 ** (attempt - 1))
                    )

        return _error(
            "ServiceNow request failed after retries",
            status_code=500,
            details=last_error,
        )

    async def get_record_by_number(
        self,
        *,
        table: str,
        number: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sysparm_query": f"number={number}",
            "sysparm_display_value": "true",
        }
        if fields:
            params["sysparm_fields"] = ",".join(fields)

        response = await self._request("GET", table, params=params)
        if not response["ok"]:
            return response

        records = response["data"] if isinstance(response["data"], list) else []
        if not records:
            return _error(
                f"No record found for number '{number}'",
                status_code=404,
            )
        return _success(records[0], status_code=response["status_code"])

    async def get_user_details(
        self,
        *,
        user_identifier: str,
        lookup_field: str = "employee_number",
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sysparm_query": f"{lookup_field}={user_identifier}",
            "sysparm_display_value": "true",
        }
        if fields:
            params["sysparm_fields"] = ",".join(fields)

        response = await self._request("GET", self.config.user_table, params=params)
        if not response["ok"]:
            return response

        users = response["data"] if isinstance(response["data"], list) else []
        if not users:
            return _error(
                f"No user found for {lookup_field}='{user_identifier}'",
                status_code=404,
            )
        return _success(users[0], status_code=response["status_code"])

    async def create_incident(self, fields: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.config.incident_table,
            params={"sysparm_display_value": "true"},
            json_body=fields,
        )

    async def update_incident(
        self,
        *,
        incident_number: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        incident_response = await self.get_record_by_number(
            table=self.config.incident_table,
            number=incident_number,
            fields=["sys_id", "number"],
        )
        if not incident_response["ok"]:
            return incident_response

        sys_id = incident_response["data"]["sys_id"]
        return await self._request(
            "PATCH",
            f"{self.config.incident_table}/{sys_id}",
            params={"sysparm_display_value": "true"},
            json_body=fields,
        )

    async def create_service_request(self, fields: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.config.request_table,
            params={"sysparm_display_value": "true"},
            json_body=fields,
        )

    async def update_service_request(
        self,
        *,
        request_number: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        request_response = await self.get_record_by_number(
            table=self.config.request_table,
            number=request_number,
            fields=["sys_id", "number"],
        )
        if not request_response["ok"]:
            return request_response

        sys_id = request_response["data"]["sys_id"]
        return await self._request(
            "PATCH",
            f"{self.config.request_table}/{sys_id}",
            params={"sysparm_display_value": "true"},
            json_body=fields,
        )


def _build_client() -> ServiceNowClient:
    return ServiceNowClient(ServiceNowConfig.from_env())


def _compact_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


@function_tool
async def get_incident_details(
    incident_number: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get ServiceNow incident details by incident number.
    """
    client = _build_client()
    return await client.get_record_by_number(
        table=client.config.incident_table,
        number=incident_number,
        fields=fields,
    )


@function_tool
async def create_incident(
    short_description: str,
    description: str,
    caller_id: str | None = None,
    urgency: str | None = None,
    impact: str | None = None,
    assignment_group: str | None = None,
    assigned_to: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a ServiceNow incident.
    """
    client = _build_client()
    payload = _compact_fields(
        {
            "short_description": short_description,
            "description": description,
            "caller_id": caller_id,
            "urgency": urgency,
            "impact": impact,
            "assignment_group": assignment_group,
            "assigned_to": assigned_to,
            "category": category,
            "subcategory": subcategory,
        }
    )
    if extra_fields:
        payload.update(extra_fields)
    return await client.create_incident(payload)


@function_tool
async def update_incident(
    incident_number: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Update a ServiceNow incident using the incident number.
    """
    client = _build_client()
    payload = _compact_fields(fields)
    if not payload:
        return _error("No incident fields provided for update", status_code=400)
    return await client.update_incident(
        incident_number=incident_number,
        fields=payload,
    )


@function_tool
async def get_user_details(
    user_identifier: str,
    lookup_field: str = "employee_number",
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get ServiceNow user details by a lookup field such as employee_number, email, or sys_id.
    """
    client = _build_client()
    return await client.get_user_details(
        user_identifier=user_identifier,
        lookup_field=lookup_field,
        fields=fields,
    )


@function_tool
async def create_service_request(
    short_description: str,
    description: str,
    requested_for: str | None = None,
    requested_by: str | None = None,
    priority: str | None = None,
    assignment_group: str | None = None,
    assigned_to: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a ServiceNow service request.
    """
    client = _build_client()
    payload = _compact_fields(
        {
            "short_description": short_description,
            "description": description,
            "requested_for": requested_for,
            "requested_by": requested_by,
            "priority": priority,
            "assignment_group": assignment_group,
            "assigned_to": assigned_to,
            "category": category,
            "subcategory": subcategory,
        }
    )
    if extra_fields:
        payload.update(extra_fields)
    return await client.create_service_request(payload)


@function_tool
async def update_service_request(
    request_number: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Update a ServiceNow service request using the request number.
    """
    client = _build_client()
    payload = _compact_fields(fields)
    if not payload:
        return _error("No service request fields provided for update", status_code=400)
    return await client.update_service_request(
        request_number=request_number,
        fields=payload,
    )


__all__ = [
    "ServiceNowClient",
    "ServiceNowConfig",
    "create_incident",
    "create_service_request",
    "get_incident_details",
    "get_user_details",
    "update_incident",
    "update_service_request",
]
