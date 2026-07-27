from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "offline_dataset_preparation" / "scripts" / "00_extract_raw.py"
DB_ENV_VARS = (
    "GEAS_DB_HOST",
    "GEAS_DB_PORT",
    "GEAS_DB_USER",
    "GEAS_DB_PASSWORD",
    "GEAS_DB_NAME",
)


def test_extract_raw_loads_db_config_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_extract_raw_module()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "GEAS_DB_HOST=example.invalid",
                "GEAS_DB_PORT=3307",
                "GEAS_DB_USER=test_user",
                "GEAS_DB_PASSWORD=test_password",
                "GEAS_DB_NAME=test_db",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = module.db_config_from_env(env_path)

    assert config["host"] == "example.invalid"
    assert config["port"] == 3307
    assert config["user"] == "test_user"
    assert config["database"] == "test_db"


def test_extract_raw_reports_missing_db_variable_names_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_extract_raw_module()
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as exc:
        module.db_config_from_env(tmp_path / "missing.env")

    message = str(exc.value)
    assert "GEAS_DB_HOST" in message
    assert "GEAS_DB_PASSWORD" in message
    assert "password=" not in message.lower()


def _load_extract_raw_module():
    module_name = "geas35_extract_raw_env_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module
    return module
