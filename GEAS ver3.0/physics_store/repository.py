from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
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


def make_record_key(
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    parts = [
        f"farm:{farm_sn}" if farm_sn is not None else "farm:any",
        f"stage:{stage_name or 'any'}",
        f"case:{data_case or 'any'}",
        f"model:{model_name or 'default'}",
    ]
    return "|".join(parts)


def make_lookup_keys(
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    data_case: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[str]:
    farm_keys = [farm_sn, None]
    stage_keys = [stage_name, None]
    case_keys = [data_case, None]
    model_keys = [model_name, "default", None]

    keys: List[str] = []
    seen = set()
    for farm in farm_keys:
        for stage in stage_keys:
            for case in case_keys:
                for model in model_keys:
                    key = make_record_key(
                        farm_sn=farm,
                        stage_name=stage,
                        data_case=case,
                        model_name=model,
                    )
                    if key not in seen:
                        keys.append(key)
                        seen.add(key)
    return keys


class PhysicsStore:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else DEFAULT_STORE_PATH

    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def save_record(
        self,
        *,
        farm_sn: Optional[int] = None,
        stage_name: Optional[str] = None,
        data_case: Optional[str] = None,
        model_name: Optional[str] = None,
        params: Dict[str, Any],
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhysicsRecord:
        key = make_record_key(
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case=data_case,
            model_name=model_name,
        )
        record = PhysicsRecord(
            key=key,
            params=dict(params),
            source=source,
            metadata=dict(metadata or {}),
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
        data_case: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[PhysicsRecord]:
        data = self._read_all()
        for key in make_lookup_keys(
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case=data_case,
            model_name=model_name,
        ):
            raw = data.get(key)
            if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
                return PhysicsRecord(
                    key=raw.get("key", key),
                    params=dict(raw.get("params", {})),
                    source=str(raw.get("source", "manual")),
                    updated_at=str(raw.get("updated_at", "")),
                    metadata=dict(raw.get("metadata", {})),
                )
        return None

    def resolve_params(
        self,
        *,
        farm_sn: Optional[int] = None,
        stage_name: Optional[str] = None,
        data_case: Optional[str] = None,
        model_name: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self.get_record(
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case=data_case,
            model_name=model_name,
        )
        if record is None:
            return dict(defaults or {})
        resolved = dict(defaults or {})
        resolved.update(record.params)
        return resolved

    def list_records(self) -> List[PhysicsRecord]:
        data = self._read_all()
        out: List[PhysicsRecord] = []
        for key, raw in data.items():
            if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
                out.append(
                    PhysicsRecord(
                        key=raw.get("key", key),
                        params=dict(raw.get("params", {})),
                        source=str(raw.get("source", "manual")),
                        updated_at=str(raw.get("updated_at", "")),
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
        out.sort(key=lambda r: r.key)
        return out
