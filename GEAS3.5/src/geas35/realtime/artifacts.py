"""Fail-closed deployment bundle validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


BUNDLE_SCHEMA_VERSION = "geas35.realtime.deployment_bundle.v1"
REQUIRED_ARTIFACT_ROLES = (
    "quality_model", "observation_scaler", "state_action_support",
    "transition_model", "ppo_policy",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    path: Path
    sha256: str
    schema_version: str | None = None


@dataclass(frozen=True)
class DeploymentBundle:
    crop: str
    manifest_path: Path
    artifacts: Mapping[str, ArtifactRecord]
    metadata: Mapping[str, Any]

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return {role: record.sha256 for role, record in self.artifacts.items()}

    def path(self, role: str) -> Path:
        if role not in self.artifacts:
            raise KeyError(f"Deployment bundle has no artifact role: {role}")
        return self.artifacts[role].path


def load_deployment_bundle(
    manifest_path: str | Path, *, require_policy: bool = True
) -> DeploymentBundle:
    manifest = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported deployment bundle schema.")
    crop = str(payload.get("crop", "")).strip().lower()
    if not crop:
        raise ValueError("Deployment bundle crop is required.")
    records = payload.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("Deployment bundle artifacts must be an object.")
    required = set(REQUIRED_ARTIFACT_ROLES)
    if not require_policy:
        required.remove("ppo_policy")
    missing = sorted(required.difference(records))
    if missing:
        raise ValueError("Deployment bundle is missing roles: " + ", ".join(missing))
    artifacts: dict[str, ArtifactRecord] = {}
    for role, raw in records.items():
        if not isinstance(raw, dict) or "path" not in raw or "sha256" not in raw:
            raise ValueError(f"Invalid artifact record: {role}")
        path = Path(str(raw["path"])).expanduser()
        path = path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Deployment artifact not found: {path}")
        expected = str(raw["sha256"]).lower()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Deployment artifact hash mismatch: {role}")
        artifacts[str(role)] = ArtifactRecord(
            role=str(role), path=path, sha256=actual,
            schema_version=(None if raw.get("schema_version") is None else str(raw["schema_version"])),
        )
    return DeploymentBundle(
        crop=crop, manifest_path=manifest, artifacts=artifacts,
        metadata=dict(payload.get("metadata") or {}),
    )


__all__ = [
    "ArtifactRecord", "BUNDLE_SCHEMA_VERSION", "DeploymentBundle",
    "REQUIRED_ARTIFACT_ROLES", "load_deployment_bundle", "sha256_file",
]
