from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_STORE_PATH = Path(__file__).with_name("physics_params_store.json")


@dataclass
class PhysicsRecord:
    key: str
    params: Dict[str, Any]
    source: str = "manual"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_time_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    try:
        return str(value)
    except Exception:
        return None


def _parse_time_value(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ts_rank(dt: Optional[datetime]) -> float:
    if dt is None:
        return float('-inf')
    return dt.timestamp()


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lambda_aliases(param_name: str) -> tuple[str, ...]:
    aliases = [str(param_name)]
    if param_name == 'eta':
        aliases.append('g_solar')
    elif param_name == 'g_solar':
        aliases.append('eta')
    if param_name in {'a0', 'a1', 'a2', 'a3'}:
        aliases.append('ACH')
    return tuple(dict.fromkeys(aliases))


def _resolve_param_lambda(metadata: Optional[Dict[str, Any]], param_name: str) -> Optional[float]:
    metadata = dict(metadata or {})
    aliases = _lambda_aliases(param_name)

    for container_key in ('lambda', 'lambdas'):
        container = metadata.get(container_key)
        if isinstance(container, dict):
            for alias in aliases:
                val = _safe_float(container.get(alias))
                if val is not None:
                    return _clamp01(val)
        else:
            val = _safe_float(container)
            if val is not None:
                return _clamp01(val)

    approval = metadata.get('approval')
    if isinstance(approval, dict):
        for container_key in ('lambda', 'lambdas'):
            container = approval.get(container_key)
            if isinstance(container, dict):
                for alias in aliases:
                    val = _safe_float(container.get(alias))
                    if val is not None:
                        return _clamp01(val)
            else:
                val = _safe_float(container)
                if val is not None:
                    return _clamp01(val)

    for alias in aliases:
        for key in (f'lambda_{alias}', f'lam_{alias}'):
            val = _safe_float(metadata.get(key))
            if val is not None:
                return _clamp01(val)
            if isinstance(approval, dict):
                val = _safe_float(approval.get(key))
                if val is not None:
                    return _clamp01(val)

    return None


def _blend_param(default_value: Any, estimated_value: Any, lam: Optional[float]) -> Any:
    if estimated_value is None:
        return default_value
    if lam is None:
        return estimated_value
    if lam <= 0.0:
        return default_value

    default_num = _safe_float(default_value)
    estimated_num = _safe_float(estimated_value)
    if default_num is not None and estimated_num is not None:
        return (1.0 - lam) * default_num + lam * estimated_num
    return estimated_value


def make_record_key(
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    model_name: Optional[str] = None,
    valid_from: Any = None,
    valid_to: Any = None,
) -> str:
    parts = [
        f"farm:{farm_sn}" if farm_sn is not None else "farm:any",
        f"stage:{stage_name or 'any'}",
        f"model:{model_name or 'default'}",
        f"from:{_normalize_time_value(valid_from) or 'any'}",
        f"to:{_normalize_time_value(valid_to) or 'any'}",
    ]
    return "|".join(parts)


def _parse_key_parts(key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in str(key).split('|'):
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        out[k] = v
    return out


def _record_scope(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    metadata = dict(raw.get('metadata', {}) or {})
    parts = _parse_key_parts(key)
    farm_val = metadata.get('farm_sn', parts.get('farm'))
    stage_val = metadata.get('stage_name', parts.get('stage'))
    model_val = metadata.get('model_name', parts.get('model'))
    valid_from = metadata.get('valid_from', parts.get('from'))
    valid_to = metadata.get('valid_to', parts.get('to'))
    data_case = metadata.get('data_case', parts.get('case'))

    if farm_val in (None, '', 'any'):
        farm_sn = None
    else:
        try:
            farm_sn = int(farm_val)
        except Exception:
            farm_sn = None

    stage_name = None if stage_val in (None, '', 'any') else str(stage_val)
    model_name = None if model_val in (None, '', 'any') else str(model_val)
    valid_from = None if valid_from in (None, '', 'any') else _normalize_time_value(valid_from)
    valid_to = None if valid_to in (None, '', 'any') else _normalize_time_value(valid_to)
    data_case = None if data_case in (None, '', 'any') else str(data_case)

    return {
        'farm_sn': farm_sn,
        'stage_name': stage_name,
        'model_name': model_name,
        'valid_from': valid_from,
        'valid_to': valid_to,
        'data_case': data_case,
    }


def _specificity_rank(record_value: Any, requested_value: Any, *, default_value: Any = None) -> Optional[int]:
    if requested_value is None:
        return 0
    if record_value == requested_value:
        return 0
    if default_value is not None and record_value == default_value:
        return 1
    if record_value is None:
        return 2
    return None


def _time_rank(scope: Dict[str, Any], target_ts: Any) -> tuple[int, float, float, float]:
    target = _parse_time_value(target_ts)
    valid_from = _parse_time_value(scope.get('valid_from'))
    valid_to = _parse_time_value(scope.get('valid_to'))

    if target is None:
        anchor = valid_to or valid_from
        return (0, 0.0, -_ts_rank(anchor), 0.0)

    if valid_from is not None and valid_to is not None and valid_from <= target <= valid_to:
        span = max((valid_to - valid_from).total_seconds(), 0.0)
        return (0, span, -_ts_rank(valid_to), -_ts_rank(valid_from))

    if valid_from is not None and valid_to is None and valid_from <= target:
        return (1, (target - valid_from).total_seconds(), -_ts_rank(valid_from), 0.0)

    if valid_to is not None and valid_to <= target:
        return (2, (target - valid_to).total_seconds(), -_ts_rank(valid_to), -_ts_rank(valid_from))

    if valid_from is None and valid_to is None:
        return (3, 0.0, 0.0, 0.0)

    future_ref = valid_from or valid_to
    if future_ref is not None:
        return (4, abs((future_ref - target).total_seconds()), _ts_rank(future_ref), 0.0)

    return (5, 0.0, 0.0, 0.0)


class PhysicsStore:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else DEFAULT_STORE_PATH

    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding='utf-8',
        )

    def save_record(
        self,
        *,
        farm_sn: Optional[int] = None,
        stage_name: Optional[str] = None,
        model_name: Optional[str] = None,
        valid_from: Any = None,
        valid_to: Any = None,
        params: Dict[str, Any],
        source: str = 'manual',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhysicsRecord:
        metadata = dict(metadata or {})
        metadata.setdefault('farm_sn', farm_sn)
        metadata.setdefault('stage_name', stage_name)
        metadata.setdefault('model_name', model_name or 'default')
        metadata.setdefault('valid_from', _normalize_time_value(valid_from))
        metadata.setdefault('valid_to', _normalize_time_value(valid_to))

        key = make_record_key(
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name=model_name,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        record = PhysicsRecord(
            key=key,
            params=dict(params),
            source=source,
            metadata=metadata,
        )
        data = self._read_all()
        data[key] = record.to_dict()
        self._write_all(data)
        return record

    def get_record(
        self,
        *,
        farm_sn: Optional[int] = None,
        stage_name: Optional[str] = None,
        model_name: Optional[str] = None,
        target_ts: Any = None,
        data_case: Optional[str] = None,
    ) -> Optional[PhysicsRecord]:
        data = self._read_all()
        best: Optional[tuple[Any, Dict[str, Any], str]] = None

        for key, raw in data.items():
            if not (isinstance(raw, dict) and isinstance(raw.get('params'), dict)):
                continue
            scope = _record_scope(raw, key)
            farm_rank = _specificity_rank(scope.get('farm_sn'), farm_sn)
            stage_rank = _specificity_rank(scope.get('stage_name'), stage_name)
            model_rank = _specificity_rank(scope.get('model_name'), model_name, default_value='default')
            if farm_rank is None or stage_rank is None or model_rank is None:
                continue

            updated_at = _parse_time_value(raw.get('updated_at'))
            time_rank = _time_rank(scope, target_ts)
            case_penalty = 0
            if data_case is not None and scope.get('data_case') not in (None, data_case):
                case_penalty = 1

            score = (
                farm_rank,
                stage_rank,
                model_rank,
                time_rank[0],
                case_penalty,
                time_rank[1],
                time_rank[2],
                time_rank[3],
                -_ts_rank(updated_at),
            )
            if best is None or score < best[0]:
                best = (score, raw, key)

        if best is None:
            return None

        raw, key = best[1], best[2]
        return PhysicsRecord(
            key=raw.get('key', key),
            params=dict(raw.get('params', {})),
            source=str(raw.get('source', 'manual')),
            updated_at=str(raw.get('updated_at', '')),
            metadata=dict(raw.get('metadata', {})),
        )

    def resolve_params(
        self,
        *,
        farm_sn: Optional[int] = None,
        stage_name: Optional[str] = None,
        model_name: Optional[str] = None,
        target_ts: Any = None,
        data_case: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self.get_record(
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name=model_name,
            target_ts=target_ts,
            data_case=data_case,
        )
        resolved = dict(defaults or {})
        if record is None:
            return resolved

        for key, estimated_value in dict(record.params).items():
            default_value = resolved.get(key)
            lam = _resolve_param_lambda(record.metadata, key)
            resolved[key] = _blend_param(default_value, estimated_value, lam)
        return resolved

    def list_records(self) -> List[PhysicsRecord]:
        data = self._read_all()
        out: List[PhysicsRecord] = []
        for key, raw in data.items():
            if isinstance(raw, dict) and isinstance(raw.get('params'), dict):
                out.append(
                    PhysicsRecord(
                        key=raw.get('key', key),
                        params=dict(raw.get('params', {})),
                        source=str(raw.get('source', 'manual')),
                        updated_at=str(raw.get('updated_at', '')),
                        metadata=dict(raw.get('metadata', {})),
                    )
                )
        out.sort(key=lambda r: (r.metadata.get('valid_from') or '', r.metadata.get('valid_to') or '', r.updated_at), reverse=True)
        return out
