# policy/profiles.py
from __future__ import annotations
from typing import Dict, Any

# Controller param presets per profile
_PROFILES: Dict[str, Dict[str, Any]] = {
    "safe_default": {
        # feature toggles
        "USE_WIND_CAP": True,
        "USE_MIN_ACH": True,
        "USE_RAMP_LIMIT": True,
        "USE_VPD_FOR_MIN_ACH": False,
        "USE_ETA_FOR_CURTAIN": False,

        # thresholds
        "WIND_CAP_TH": 5.0,
        "WIND_CAP_OPEN": 20,
        "ACH_MIN_DAY": 0.10,
        "ACH_MIN_NIGHT": 0.05,
        "RAMP_LIMIT": 15,
    },
}

def get_profile(name: str | None) -> Dict[str, Any]:
    """Return controller param preset for a given profile name."""
    if not name:
        return {}
    return _PROFILES.get(name, {})
