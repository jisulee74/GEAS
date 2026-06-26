from typing import Any, Dict, Optional, Tuple


def map_controller_output_to_legacy(
    control_res: Dict[str, Any],
) -> Dict[str, Any]:

    legacy: Dict[str, Any] = {}

    # --- WINDOW ---
    window: Optional[Tuple[Any, Any, Any]] = control_res.get("window")
    if window:
        pct_final, _, _ = window
        try:
            pct = int(pct_final)
        except (TypeError, ValueError):
            pct = 0

        legacy["window"] = (0, "both", pct)

    # --- CURTAIN ---
    curtain: Optional[Tuple[Any, Any]] = control_res.get("curtain")
    if curtain:
        mode, size = curtain
        try:
            pct = int(size)
        except (TypeError, ValueError):
            pct = 0

        legacy_mode = "cha_gwang"
        legacy["curtain"] = (legacy_mode, pct)

    # --- FCU ---
    fcu: Optional[Tuple[Any, Any]] = control_res.get("fcu")
    if fcu:
        mode, state = fcu

        if state == "off":
            legacy["fcu"] = ("off", 0)
        elif mode == "cool":
            legacy["fcu"] = ("cooling", 1)
        elif mode == "heat":
            legacy["fcu"] = ("heating", 1)
        else:
            legacy["fcu"] = ("off", 0)

    # --- FAN ---
    fan: Optional[Tuple[Any]] = control_res.get("fan")
    if fan:
        state = fan[0]
        if state == "on":
            legacy["fan"] = 1
            
        elif state == "off":
            legacy["fan"] = 0

    return legacy
