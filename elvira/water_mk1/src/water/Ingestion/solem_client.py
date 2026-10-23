"""
 * author Antonio Sirignano
 * created on 23-10-2026-15h-50m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

import json
import os
import re
import time
import requests

from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv

DEFAULT_BASE_URL = 'https://mysolem.com'

_USERID_RE = re.compile(r'userId\s*=\s*["\']([a-f0-9]{24})["\']')
_MODULE_MARKER = "let module = "
_INPUTS_MARKER = "var inputs = "
_SENSOR_NAME_TAG_RE = re.compile(r"<input[^>]*\bname=\"sensor-name\"[^>]*>")
_VALUE_ATTR_RE = re.compile(r"\bvalue=\"([^\"]*)\"")
_INPUT_ID_ATTR_RE = re.compile(r"\bdata-input-id=\"([a-f0-9]{24})\"")
_EXPRESSION_RE = re.compile(r"^x(?:\s*([*/])\s*(\d+(?:\.\d+)?))?$")

FLOW_METER_TYPE = 1

class SolemError(Exception):
    """Generic Error"""

class SolemAuthError(SolemError):
    """Authentication Error"""

class SolemConnectionError(SolemError):
    """Correction Error"""

#   EXTRACTORS -------------------------------------------------------------------------------------

def apply_expression(expression: str | None, value: float) -> Any | None:
    if not expression:
        return float(value)
    match = _EXPRESSION_RE.match(expression.strip())
    if match is None:
        return None
    operator, operand = match.groups()
    if operator is None:
        return float(value)
    number = float(operand)
    if operator == '*':
        return round(value * number, 3)
    if number == 0:
        return None
    return round(value / number, 3)

def _extract_js_literal(html: str, marker: str, opener: str) -> Any | None:
    closer = {'{': '}', '[': ']'}[opener]
    at = html.find(marker)
    if at == -1:
        return None
    start = html.find(opener, at)
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(html)):
        ch = html[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start: i+1])
                except json.JSONDecodeError:
                    return None
    return None

def _extract_input_labels(html: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for tag in _SENSOR_NAME_TAG_RE.findall(html):
        input_id, value = _INPUT_ID_ATTR_RE.search(tag), _VALUE_ATTR_RE.search(tag)
        if input_id and value and value.group(1).strip():
            labels[input_id.group(1)] = value.group(1).strip()
    return labels

def _js_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError('It\'s necessary a datetime with time zone (tzinfo)')
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

#   CLIENT -----------------------------------------------------------------------------------------

class SolemClient:
    def __init__(self,
                 email: str | None = None,
                 password: str | None = None,
                 region: str | None = None,
                 base_url: str | None = None,
                 pause_s: float = 0.5,
                 timeout: float = 30.0,
                 ):
        load_dotenv()
        self._email = email or os.getenv("SOLEM_EMAIL")
        self._password = password or os.getenv("SOLEM_PASSWORD")
        self._region = region or os.getenv("SOLEM_REGION", "Europe")
        self._base = (base_url or os.getenv("SOLEM_BASE_URL") or DEFAULT_BASE_URL).rstrip('/')
        if not self._email or not self._password:
            raise ValueError("Set SOLEM_EMAIL and SOLEM_PASSWORD in environment file")
        self.pause_s = pause_s
        self.timeout = timeout
        self.session = requests.Session()
        self.user_id: str | None = None

    def login(self) -> str:
        payload = {
            'email': self._email,
            'password': self._password,
            'rememberMe': 'on',
            'country-select': self._region
        }

        try:
            resp = self.session.post(f'{self._base}/login',
                                     data=payload,
                                     timeout=self.timeout
                                     )
        except requests.RequestException as err:
            raise SolemConnectionError(f'Login failed: {err}') from err

        user_id = self._parse_user_id(resp.text) or self._fetch_user_id()
        if user_id is None:
            raise SolemAuthError('MySolem Credential not valid')
        self.user_id = user_id
        return user_id

    @staticmethod
    def _parse_user_id(html: str) -> str | None:
        m = _USERID_RE.search(html)
        return m.group(1) if m else None

    def _fetch_user_id(self) -> str | None:
        try:
            resp = self.session.get(f'{self._base}/modules', timeout=self.timeout)
        except requests.RequestException as err:
            raise SolemConnectionError(f'Impossible to read /modules: {err}') from err
        return self._parse_user_id(resp.text)

    @staticmethod
    def _logged_out(resp: requests.Response) -> bool:
        return resp.url.split('?')[0].rstrip('/').endswith('/login')

    def _request(self, method: str, path: str, _retry: bool = True, **kwargs: Any) -> requests.Response:
        if self.user_id is None:
            self.login()
        time.sleep(self.pause_s)
        try:
            resp = self.session.request(method, f'{self._base}{path}', timeout=self.timeout, **kwargs)
        except requests.RequestException as err:
            raise SolemConnectionError(f'{method} {path} failed: {err}') from err

        if self._logged_out(resp) or resp.status_code in (401, 403):
            if not _retry:
                raise SolemAuthError('Session exprired or new login failed')
            self.user_id = None
            self.login()
            return self._request(method, path, _retry=False, **kwargs)

        try:
            resp.raise_for_status()
        except requests.HTTPError as err:
            raise SolemConnectionError(f'{method} {path} failed: {err}') from err
        return resp

    def _json(self, method: str,  path, **kwargs: Any) -> Any:
        try:
            return self._request(method, path, **kwargs).json()
        except ValueError as err:
            raise SolemConnectionError(f'{method} {path} did not return JSON') from err

    def get_module_ids(self) -> list[str]:
        data = self._json('POST', f'/user/{self.user_id}/modules')
        modules = data.get('modules', []) if isinstance(data, dict) else []
        return [m['id'] for m in modules if isinstance(m, dict) and m.get('id')]

    def get_module_page(self, module_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        html = self._request('GET', f'/module/{self.user_id}').text
        module = _extract_js_literal(html, _MODULE_MARKER, '{')
        labels = _extract_input_labels(html)
        raw_inputs = _extract_js_literal(html, _INPUTS_MARKER, '[') or []
        inputs = [{**r, 'name': r.get('name') or labels.get(r['id'], '')} for r in raw_inputs if isinstance(r, dict) and r.get('id')]
        return (module if isinstance(module, dict) else {}), inputs

    @staticmethod
    def flow_inputs(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [i for i in inputs if i.get('type') == FLOW_METER_TYPE]

    def get_last_tick(self, input_id: str) -> dict[str, Any]:
        data = self._json('GET', '/input/getLastTickFromInput', params={'inputId': input_id})
        tick = data.get('tick') if isinstance(data, dict) else None
        return tick if isinstance(tick, dict) else {}

    def get_sensor_data(self, module_id: str, start: datetime, end: datetime) -> Any:
        return self._json('GET', '/remote/module/getComputedSensorData', params={'moduleId': module_id,
                                                                                 'startData': _js_iso(start),
                                                                                 'endData': _js_iso(end),
                                                                                 'forcedRawOrComputed': 'computed',
                                                                                 })

