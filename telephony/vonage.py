from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import requests


BASE_URL = "https://rest.nexmo.com"
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(slots=True)
class VonageAvailableNumber:
    country: str
    msisdn: str
    features: list[str]
    number_type: str
    cost: str | None = None
    monthly_rental_cost: str | None = None
    setup_cost: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["e164"] = f"+{self.msisdn.lstrip('+')}"
        return payload


@dataclass(slots=True)
class VonageOwnedNumber:
    country: str
    msisdn: str
    features: list[str]
    number_type: str
    voice_callback_type: str | None = None
    voice_callback_value: str | None = None
    app_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["e164"] = f"+{self.msisdn.lstrip('+')}"
        return payload


class VonageAPIError(RuntimeError):
    pass


class VonageClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = (api_key or os.getenv("VONAGE_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.getenv("VONAGE_API_SECRET", "")).strip()
        self.timeout_seconds = timeout_seconds

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Vonage credentials are missing. Set VONAGE_API_KEY and VONAGE_API_SECRET."
            )

    def search_available_numbers(
        self,
        *,
        country: str,
        number_type: str | None = None,
        features: list[str] | None = None,
        pattern: str | None = None,
        search_pattern: int | None = None,
        size: int = 20,
    ) -> list[VonageAvailableNumber]:
        params: dict[str, Any] = {"country": country.upper()}
        if number_type:
            params["type"] = number_type
        if features:
            params["features"] = ",".join(features)
        if pattern:
            params["pattern"] = pattern
        if search_pattern is not None:
            params["search_pattern"] = str(search_pattern)
        if size > 0:
            params["size"] = str(size)

        payload = self._request("GET", "/number/search", params=params)
        numbers = payload.get("numbers", [])
        return [self._parse_available_number(item, country.upper()) for item in numbers]

    def list_owned_numbers(
        self,
        *,
        pattern: str | None = None,
        search_pattern: int | None = None,
        size: int = 100,
    ) -> list[VonageOwnedNumber]:
        params: dict[str, Any] = {}
        if pattern:
            params["pattern"] = pattern
        if search_pattern is not None:
            params["search_pattern"] = str(search_pattern)
        if size > 0:
            params["size"] = str(size)

        payload = self._request("GET", "/account/numbers", params=params)
        numbers = payload.get("numbers", [])
        return [self._parse_owned_number(item) for item in numbers]

    def buy_number(self, *, country: str, msisdn: str) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/number/buy",
            data={"country": country.upper(), "msisdn": self._normalize_msisdn(msisdn)},
        )
        return dict(payload)

    def update_number(
        self,
        *,
        country: str,
        msisdn: str,
        voice_callback_type: str,
        voice_callback_value: str,
        app_id: str | None = None,
        voice_status_callback: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "country": country.upper(),
            "msisdn": self._normalize_msisdn(msisdn),
            "voiceCallbackType": voice_callback_type,
            "voiceCallbackValue": voice_callback_value,
        }
        if app_id:
            data["app_id"] = app_id
        if voice_status_callback:
            data["voiceStatusCallback"] = voice_status_callback

        payload = self._request("POST", "/number/update", data=data)
        return dict(payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = requests.request(
            method=method,
            url=f"{BASE_URL}{path}",
            params=params,
            data=data,
            auth=(self.api_key, self.api_secret),
            timeout=self.timeout_seconds,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise VonageAPIError(
                f"Vonage request to {path} returned non-JSON response: HTTP {response.status_code}."
            ) from exc

        error_code = str(payload.get("error-code", "0"))
        if response.status_code >= 400 or error_code not in {"0", "200"}:
            label = payload.get("error-code-label") or payload.get("error-code-labels") or payload
            raise VonageAPIError(f"Vonage request failed for {path}: {label}")

        return payload

    @staticmethod
    def _normalize_msisdn(msisdn: str) -> str:
        value = msisdn.strip()
        if value.startswith("+"):
            value = value[1:]
        if not value.isdigit():
            raise ValueError("Vonage MSISDN must contain digits only, optionally prefixed with '+'.")
        return value

    @classmethod
    def _parse_available_number(cls, item: dict[str, Any], country: str) -> VonageAvailableNumber:
        return VonageAvailableNumber(
            country=str(item.get("country") or country).upper(),
            msisdn=cls._normalize_msisdn(str(item.get("msisdn", ""))),
            features=cls._coerce_features(item.get("features")),
            number_type=str(item.get("type") or item.get("number_type") or ""),
            cost=cls._string_or_none(item.get("cost")),
            monthly_rental_cost=cls._string_or_none(item.get("monthlyRentalCost")),
            setup_cost=cls._string_or_none(item.get("setupCost")),
        )

    @classmethod
    def _parse_owned_number(cls, item: dict[str, Any]) -> VonageOwnedNumber:
        return VonageOwnedNumber(
            country=str(item.get("country") or "").upper(),
            msisdn=cls._normalize_msisdn(str(item.get("msisdn", ""))),
            features=cls._coerce_features(item.get("features")),
            number_type=str(item.get("type") or item.get("number_type") or ""),
            voice_callback_type=cls._string_or_none(item.get("voiceCallbackType")),
            voice_callback_value=cls._string_or_none(item.get("voiceCallbackValue")),
            app_id=cls._string_or_none(item.get("app_id")),
        )

    @staticmethod
    def _coerce_features(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item).strip()]
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(",") if part.strip()]
        return []

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
