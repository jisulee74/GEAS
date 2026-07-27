# Growth Stage Calculation

GEAS3.5 calculates crop growth stage from `crop_info.trans_crop_date`.

## Source Table

Realtime and historical lookup should use `crop_info`:

```sql
SELECT *
FROM crop_info
WHERE iot_data_idx = ?
  AND COALESCE(del_yn, 'N') = 'N'
  AND trans_crop_date IS NOT NULL
  AND trans_crop_date <= ?
  AND (crop_end_date IS NULL OR crop_end_date >= ?)
ORDER BY trans_crop_date DESC, idx DESC
LIMIT 1;
```

For `iot_data_idx=97`, the current active row on 2026-07-14 is cucumber
with `trans_crop_date=2026-06-16`.

## Rules

The DAT rule table lives at `configs/crops/growth_stage_rules.csv`.

DAT is one-based. The transplant calendar date is DAT 1.

Temperature ranges are preserved, and the controller target is the midpoint of
each day/night range. If a source guideline is an upper bound rather than a
closed range, the available bound is used as the nominal target and the note
column records the interpretation.

Policy decisions:

- Melon stays in the final maturity/harvest stage after DAT 75.
- Cucumber stays in the late-harvest stage after DAT 70.
- Cucumber main harvest keeps separate night ranges: early night 15-18 C,
  late night 13-15 C. The compatibility night range is stored as 13-18 C.
- Cucumber main harvest uses the clear-day daytime range 25-28 C by default.
  Partly cloudy and cloudy days use the cloudy-day daytime range 20-25 C.
- Cucumber late-harvest targets use the lowered ranges: day 23-26 C,
  early night 13-16 C, late night 11-13 C. The compatibility night range is
  stored as 11-16 C, with the split ranges preserved separately.

## Solar Period

`geas35.core.solar_time` calculates sunrise and sunset in the same style as the
backup controller:

- If `pvlib` is installed, it samples solar position every 15 minutes and treats
  positive apparent elevation as daylight.
- If `pvlib` is unavailable, it falls back to 07:00 sunrise and 18:00 sunset in
  local time.

Night is split as follows:

- `early_night`: from sunset through the next 6 hours.
- `late_night`: after `early_night` until sunrise.

`GrowthStage.effective_temperature_target(...)` combines the current growth
stage with this solar-period classifier. For controller compatibility,
`GrowthStage.to_controller_stage_config()` still returns the overall midpoint
when called without solar arguments. When called with `target_ts`, `lat`, and
`lon`, it also returns:

- `solar_period`
- `current_target_temp_c`
- `current_temp_range_c`

For cucumber, this means the night target changes by period:

| Stage | Period | Allowed range C | Midpoint target C |
| --- | --- | ---: | ---: |
| main_harvest | early_night | 15-18 | 16.5 |
| main_harvest | late_night | 13-15 | 14.0 |
| late_harvest | early_night | 13-16 | 14.5 |
| late_harvest | late_night | 11-13 | 12.0 |

## Clear/Cloudy Day Handling

The cucumber main-harvest source guideline distinguishes clear-day daytime
temperature from cloudy-day daytime temperature. The current rule file stores
both ranges:

- clear day: 25-28 C
- partly cloudy day: 20-25 C
- cloudy day: 20-25 C

The main `data_silla_enc` table does not have a direct clear/cloudy weather
label. It does have `out_light`, `out_light_sum`, `out_rainfall`, and
`out_rain`, so it remains a fallback source when weather-station data is absent.

The implemented primary method uses WMO-style sunshine ratio from
`weather_data.sun_Time`:

- `iot_data_info` links `iot_data_idx=97` to weather station `742290A001`
  (`stn_nm='Sangju-si Chosan-dong'`).
- `weather_data.sun_Time` is treated as daily accumulated sunshine minutes.
  Daily sunshine minutes are calculated as `MAX(sun_Time)` for the target date,
  ignoring non-numeric values such as `'-'`.
- Theoretical possible sunshine duration is calculated from latitude/longitude
  sunrise and sunset.
- Sunshine ratio is
  `daily_sunshine_minutes / possible_sunshine_minutes * 100`.

Classification:

| Sunshine ratio | Daylight condition | Daytime range for cucumber main harvest |
| ---: | --- | ---: |
| >= 70% | sunny | 25-28 C |
| > 30% and < 70% | partly_cloudy | 20-25 C |
| <= 30% | cloudy | 20-25 C |

Coverage note: iot 97 has `weather_data` records for station `742290A001`
from `2024-10-30 15:30` onward, while greenhouse sensor data begins at
`2024-03-05 10:50`. For dates before weather-station coverage, or dates where
`sun_Time` has no numeric values, `geas35.data.weather` returns
`daylight_condition='unknown'`. Growth-stage temperature selection treats
`unknown` as the existing clear-day default range rather than inventing a
cloudy-day label.

Use `geas35.data.fetch_growth_stage_controller_config_at_from_db(...)` when the
runtime needs a single DB-backed call that combines crop growth stage,
sunshine-ratio daylight condition, solar period, and controller temperature
range.

## Offline Dataset Columns

Offline parquet preparation adds two compact numeric columns during
`4_preprocessed/1_input_schema_prepared`:

- `growth_stage_order`: integer stage number from the crop DAT rule table.
- `growth_stage_dat`: one-based days after transplanting.

`growth_stage_name` is not written to parquet by default. Stage names remain in
`configs/crops/growth_stage_rules.csv`, and the offline manifest records the
rule-table path plus SHA-256 hash so the integer labels can be resolved later.

For `datasets/iot97_historical`, the offline runner first reads
`1_raw/manifest.csv` and uses each `series_id` row's `actual_start` as the
series-level transplant/start date. If that manifest is unavailable, the helper
falls back to the minimum `reg_date` in each series, which is useful for tests
but should not be treated as the preferred production source.

The shared implementation lives in
`geas35.preprocessing.add_growth_stage_columns(...)`. Offline scripts call this
function when writing parquet, and realtime code can use the same rule table and
DAT logic through either this helper or the DB-backed functions under
`geas35.data`.

Other weather-side candidates exist but are not the primary implementation:

- `weather_data.sun_Qy` stores solar radiation and can be used as a fallback or
  validation signal.
- `openapi_get_fmland_vilage_fcst.sky`, `openapi_asos_hourly_info.dc10Tca`,
  `openapi_asos_hourly_info.dc10LmcsCa`, `openapi_asos_daily_info.avgTca`, and
  `openapi_asos_daily_info.avgLmac` are structurally useful, but those OpenAPI
  tables were empty in the checked DB snapshot.
