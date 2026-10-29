"""Input schema, unit canonicalization, and quality-control preprocessing."""

from importlib import import_module

_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
ACTION_COLUMNS = _input_schema.ACTION_COLUMNS
EXTERNAL_STATE_COLUMNS = _input_schema.EXTERNAL_STATE_COLUMNS
FEATURE_COLUMNS = _input_schema.FEATURE_COLUMNS
INTERNAL_STATE_COLUMNS = _input_schema.INTERNAL_STATE_COLUMNS
STATE_COLUMNS = _input_schema.STATE_COLUMNS
SYSTEM_STATE_COLUMNS = _input_schema.SYSTEM_STATE_COLUMNS
TIME_COLUMN = _input_schema.TIME_COLUMN
NUMERIC_FEATURE_COLUMNS = _input_schema.NUMERIC_FEATURE_COLUMNS
coerce_geas_numeric_features = _input_schema.coerce_geas_numeric_features
coerce_numeric_columns = _input_schema.coerce_numeric_columns
ensure_columns = _input_schema.ensure_columns
ensure_geas_feature_columns = _input_schema.ensure_geas_feature_columns
parse_reg_date = _input_schema.parse_reg_date
prepare_geas_input_schema = _input_schema.prepare_geas_input_schema
standardize_geas_input_schema = _input_schema.standardize_geas_input_schema

_unit = import_module("geas35.preprocessing.2_unit_canonicalization")
BINARY_ACTION_COLUMNS = _unit.BINARY_ACTION_COLUMNS
DERIVED_RENAME_MAP = _unit.DERIVED_RENAME_MAP
HUMIDITY_PERCENT_COLUMNS = _unit.HUMIDITY_PERCENT_COLUMNS
NON_NEGATIVE_COLUMNS = _unit.NON_NEGATIVE_COLUMNS
PERCENT_ACTION_COLUMNS = _unit.PERCENT_ACTION_COLUMNS
canonicalize_feature_units = _unit.canonicalize_feature_units
canonicalize_for_derived = _unit.canonicalize_for_derived
normalize_feature_ranges = _unit.normalize_feature_ranges
normalize_for_derived = _unit.normalize_for_derived

_quality_stage = import_module("geas35.preprocessing.3_missing_outliers_handling")
ACTION_CONTROL_LOG_MAP = _quality_stage.ACTION_CONTROL_LOG_MAP
AI_BASED_OUTLIER_COLUMNS = _quality_stage.AI_BASED_OUTLIER_COLUMNS
RULE_BASED_OUTLIER_COLUMNS = _quality_stage.RULE_BASED_OUTLIER_COLUMNS
ACTION_RESTORED_FLAG_SUFFIX = _quality_stage.ACTION_RESTORED_FLAG_SUFFIX
AI_ANOMALY_SCORE_SUFFIX = _quality_stage.AI_ANOMALY_SCORE_SUFFIX
AI_CONFIDENCE_SUFFIX = _quality_stage.AI_CONFIDENCE_SUFFIX
AI_OUTLIER_FLAG_SUFFIX = _quality_stage.AI_OUTLIER_FLAG_SUFFIX
CONTROL_LOG_TIME_CANDIDATES = _quality_stage.CONTROL_LOG_TIME_CANDIDATES
DEFAULT_DOMAIN_RANGE_PATH = _quality_stage.DEFAULT_DOMAIN_RANGE_PATH
DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD = (
    _quality_stage.DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD
)
DEFAULT_RULE_AGGREGATE_FLAG = _quality_stage.DEFAULT_RULE_AGGREGATE_FLAG
DomainRangeRule = _quality_stage.DomainRangeRule
INVALID_FLAG_SUFFIX = _quality_stage.INVALID_FLAG_SUFFIX
MISSING_FLAG_SUFFIX = _quality_stage.MISSING_FLAG_SUFFIX
OBSERVATION_RULE_COLUMNS = _quality_stage.OBSERVATION_RULE_COLUMNS
QUALITY_AI_OUTLIER_FLAG_COLUMN = _quality_stage.QUALITY_AI_OUTLIER_FLAG_COLUMN
QUALITY_IMPUTED_FLAG_COLUMN = _quality_stage.QUALITY_IMPUTED_FLAG_COLUMN
QUALITY_IMPUTED_FLAG_SUFFIX = _quality_stage.QUALITY_IMPUTED_FLAG_SUFFIX
QUALITY_INVALID_FLAG_COLUMN = _quality_stage.QUALITY_INVALID_FLAG_COLUMN
RESAMPLED_ROW_COLUMN = _quality_stage.RESAMPLED_ROW_COLUMN
SEGMENT_COLUMN = _quality_stage.SEGMENT_COLUMN
action_restored_flag_column = _quality_stage.action_restored_flag_column
add_missing_flags = _quality_stage.add_missing_flags
ai_anomaly_score_column = _quality_stage.ai_anomaly_score_column
ai_confidence_column = _quality_stage.ai_confidence_column
apply_default_domain_flags = _quality_stage.apply_default_domain_flags
apply_default_rule_based_outlier_flags = (
    _quality_stage.apply_default_rule_based_outlier_flags
)
apply_domain_range_flags = _quality_stage.apply_domain_range_flags
apply_rule_based_outlier_flags = _quality_stage.apply_rule_based_outlier_flags
apply_smartfarm_korea_domain_flags = _quality_stage.apply_smartfarm_korea_domain_flags
invalid_flag_column = _quality_stage.invalid_flag_column
load_domain_range_rules = _quality_stage.load_domain_range_rules
load_smartfarm_korea_domain_rules = _quality_stage.load_smartfarm_korea_domain_rules
missing_flag_column = _quality_stage.missing_flag_column
normalize_missing_input = _quality_stage.normalize_missing_input
prepare_missing_features = _quality_stage.prepare_missing_features
prepare_missing_outliers_handled_features = (
    _quality_stage.prepare_missing_outliers_handled_features
)
prepare_quality_features = _quality_stage.prepare_quality_features
quality_ai_outlier_flag_column = _quality_stage.quality_ai_outlier_flag_column
quality_imputed_flag_column = _quality_stage.quality_imputed_flag_column
raw_value_column = _quality_stage.raw_value_column
restore_action_columns_from_control_log = (
    _quality_stage.restore_action_columns_from_control_log
)
rule_outlier_flag_column = _quality_stage.rule_outlier_flag_column

_growth_stage = import_module("geas35.preprocessing.growth_stage_features")
GROWTH_STAGE_DAT_COLUMN = _growth_stage.GROWTH_STAGE_DAT_COLUMN
GROWTH_STAGE_NAME_COLUMN = _growth_stage.GROWTH_STAGE_NAME_COLUMN
GROWTH_STAGE_ORDER_COLUMN = _growth_stage.GROWTH_STAGE_ORDER_COLUMN
add_growth_stage_columns = _growth_stage.add_growth_stage_columns
add_growth_stage_columns_from_crop_cycles = (
    _growth_stage.add_growth_stage_columns_from_crop_cycles
)
CROP_CYCLE_ID_COLUMN = _growth_stage.CROP_CYCLE_ID_COLUMN
EFFECTIVE_CROP_END_DATE_COLUMN = _growth_stage.EFFECTIVE_CROP_END_DATE_COLUMN
GROWTH_STAGE_UNMATCHED_FLAG_COLUMN = _growth_stage.GROWTH_STAGE_UNMATCHED_FLAG_COLUMN
TRANSPLANT_DATE_COLUMN = _growth_stage.TRANSPLANT_DATE_COLUMN
growth_stage_rules_sha256 = _growth_stage.growth_stage_rules_sha256

__all__ = [
    "CROP_CYCLE_ID_COLUMN",
    "EFFECTIVE_CROP_END_DATE_COLUMN",
    "GROWTH_STAGE_UNMATCHED_FLAG_COLUMN",
    "TRANSPLANT_DATE_COLUMN",
    "add_growth_stage_columns_from_crop_cycles",
    "ACTION_COLUMNS",
    "ACTION_CONTROL_LOG_MAP",
    "AI_BASED_OUTLIER_COLUMNS",
    "ACTION_RESTORED_FLAG_SUFFIX",
    "AI_ANOMALY_SCORE_SUFFIX",
    "AI_CONFIDENCE_SUFFIX",
    "AI_OUTLIER_FLAG_SUFFIX",
    "BINARY_ACTION_COLUMNS",
    "CONTROL_LOG_TIME_CANDIDATES",
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DERIVED_RENAME_MAP",
    "DomainRangeRule",
    "EXTERNAL_STATE_COLUMNS",
    "FEATURE_COLUMNS",
    "GROWTH_STAGE_DAT_COLUMN",
    "GROWTH_STAGE_NAME_COLUMN",
    "GROWTH_STAGE_ORDER_COLUMN",
    "HUMIDITY_PERCENT_COLUMNS",
    "INTERNAL_STATE_COLUMNS",
    "INVALID_FLAG_SUFFIX",
    "MISSING_FLAG_SUFFIX",
    "NON_NEGATIVE_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "OBSERVATION_RULE_COLUMNS",
    "PERCENT_ACTION_COLUMNS",
    "QUALITY_AI_OUTLIER_FLAG_COLUMN",
    "QUALITY_IMPUTED_FLAG_COLUMN",
    "QUALITY_IMPUTED_FLAG_SUFFIX",
    "QUALITY_INVALID_FLAG_COLUMN",
    "RESAMPLED_ROW_COLUMN",
    "RULE_BASED_OUTLIER_COLUMNS",
    "SEGMENT_COLUMN",
    "STATE_COLUMNS",
    "SYSTEM_STATE_COLUMNS",
    "TIME_COLUMN",
    "action_restored_flag_column",
    "add_growth_stage_columns",
    "add_missing_flags",
    "ai_anomaly_score_column",
    "ai_confidence_column",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "canonicalize_feature_units",
    "canonicalize_for_derived",
    "coerce_geas_numeric_features",
    "coerce_numeric_columns",
    "ensure_columns",
    "ensure_geas_feature_columns",
    "growth_stage_rules_sha256",
    "invalid_flag_column",
    "load_domain_range_rules",
    "load_smartfarm_korea_domain_rules",
    "missing_flag_column",
    "normalize_feature_ranges",
    "normalize_for_derived",
    "normalize_missing_input",
    "parse_reg_date",
    "prepare_geas_input_schema",
    "prepare_missing_features",
    "prepare_missing_outliers_handled_features",
    "prepare_quality_features",
    "quality_ai_outlier_flag_column",
    "quality_imputed_flag_column",
    "raw_value_column",
    "restore_action_columns_from_control_log",
    "rule_outlier_flag_column",
    "standardize_geas_input_schema",
]
