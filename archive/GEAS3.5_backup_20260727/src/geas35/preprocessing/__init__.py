"""Input schema, missing-value, unit, and outlier handling logic."""

from importlib import import_module

_input_schema = import_module("geas35.preprocessing.1_input_schema_preparation")
NUMERIC_FEATURE_COLUMNS = _input_schema.NUMERIC_FEATURE_COLUMNS
coerce_geas_numeric_features = _input_schema.coerce_geas_numeric_features
coerce_numeric_columns = _input_schema.coerce_numeric_columns
ensure_columns = _input_schema.ensure_columns
ensure_geas_feature_columns = _input_schema.ensure_geas_feature_columns
parse_reg_date = _input_schema.parse_reg_date
prepare_geas_input_schema = _input_schema.prepare_geas_input_schema
standardize_geas_input_schema = _input_schema.standardize_geas_input_schema

_missing = import_module("geas35.preprocessing.2_missing_values_handling")
ACTION_COLUMNS = _missing.ACTION_COLUMNS
ACTION_RESTORED_FLAG_SUFFIX = _missing.ACTION_RESTORED_FLAG_SUFFIX
EXTERNAL_STATE_COLUMNS = _missing.EXTERNAL_STATE_COLUMNS
FEATURE_COLUMNS = _missing.FEATURE_COLUMNS
INTERNAL_STATE_COLUMNS = _missing.INTERNAL_STATE_COLUMNS
MISSING_FLAG_SUFFIX = _missing.MISSING_FLAG_SUFFIX
RESAMPLED_ROW_COLUMN = _missing.RESAMPLED_ROW_COLUMN
SEGMENT_COLUMN = _missing.SEGMENT_COLUMN
STATE_COLUMNS = _missing.STATE_COLUMNS
SYSTEM_STATE_COLUMNS = _missing.SYSTEM_STATE_COLUMNS
TIME_COLUMN = _missing.TIME_COLUMN
action_restored_flag_column = _missing.action_restored_flag_column
add_missing_flags = _missing.add_missing_flags
missing_flag_column = _missing.missing_flag_column
normalize_missing_input = _missing.normalize_missing_input
prepare_missing_features = _missing.prepare_missing_features
restore_action_columns_from_control_log = _missing.restore_action_columns_from_control_log

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

_outlier = import_module("geas35.preprocessing.4_outlier_handling")
DEFAULT_AGGREGATE_FLAG = _outlier.DEFAULT_AGGREGATE_FLAG
DEFAULT_DOMAIN_RANGE_PATH = _outlier.DEFAULT_DOMAIN_RANGE_PATH
DEFAULT_RULE_AGGREGATE_FLAG = _outlier.DEFAULT_RULE_AGGREGATE_FLAG
DEFAULT_RULE_FLAG_PREFIX = _outlier.DEFAULT_RULE_FLAG_PREFIX
DEFAULT_AI_OUTLIER_FLAG_COLUMN = _outlier.DEFAULT_AI_OUTLIER_FLAG_COLUMN
DEFAULT_AI_WARNING_FLAG_COLUMN = _outlier.DEFAULT_AI_WARNING_FLAG_COLUMN
DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN = _outlier.DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN
DEFAULT_IMPUTED_FLAG_COLUMN = _outlier.DEFAULT_IMPUTED_FLAG_COLUMN
DEFAULT_TCN_FLAG_COLUMN = _outlier.DEFAULT_TCN_FLAG_COLUMN
DEFAULT_TCN_PRED_VALUE_COLUMN = _outlier.DEFAULT_TCN_PRED_VALUE_COLUMN
DEFAULT_TCN_SCORE_COLUMN = _outlier.DEFAULT_TCN_SCORE_COLUMN
DEFAULT_TCN_WARNING_FLAG_COLUMN = _outlier.DEFAULT_TCN_WARNING_FLAG_COLUMN
DomainRangeRule = _outlier.DomainRangeRule
OBSERVATION_RULE_COLUMNS = _outlier.OBSERVATION_RULE_COLUMNS
TCNOutlierConfig = _outlier.TCNOutlierConfig
ai_outlier_flag_column = _outlier.ai_outlier_flag_column
ai_warning_flag_column = _outlier.ai_warning_flag_column
apply_controller_value_policy = _outlier.apply_controller_value_policy
apply_default_domain_flags = _outlier.apply_default_domain_flags
apply_default_rule_based_outlier_flags = (
    _outlier.apply_default_rule_based_outlier_flags
)
apply_domain_range_flags = _outlier.apply_domain_range_flags
apply_outlier_flags = _outlier.apply_outlier_flags
apply_rule_based_outlier_flags = _outlier.apply_rule_based_outlier_flags
apply_smartfarm_korea_domain_flags = _outlier.apply_smartfarm_korea_domain_flags
apply_tcn_outlier_flags = _outlier.apply_tcn_outlier_flags
controller_value_column = _outlier.controller_value_column
high_confidence_outlier_flag_column = _outlier.high_confidence_outlier_flag_column
load_domain_range_rules = _outlier.load_domain_range_rules
load_smartfarm_korea_domain_rules = _outlier.load_smartfarm_korea_domain_rules
outlier_imputed_flag_column = _outlier.imputed_flag_column
raw_value_column = _outlier.raw_value_column
rule_outlier_flag_column = _outlier.rule_outlier_flag_column
tcn_outlier_flag_column = _outlier.tcn_outlier_flag_column
tcn_pred_value_column = _outlier.tcn_pred_value_column
tcn_score_column = _outlier.tcn_score_column
tcn_warning_flag_column = _outlier.tcn_warning_flag_column

_quality_stage = import_module("geas35.preprocessing.3_missing_outliers_handling")
AI_ANOMALY_SCORE_SUFFIX = _quality_stage.AI_ANOMALY_SCORE_SUFFIX
AI_CONFIDENCE_SUFFIX = _quality_stage.AI_CONFIDENCE_SUFFIX
AI_OUTLIER_FLAG_SUFFIX = _quality_stage.AI_OUTLIER_FLAG_SUFFIX
INVALID_FLAG_SUFFIX = _quality_stage.INVALID_FLAG_SUFFIX
QUALITY_AI_OUTLIER_FLAG_COLUMN = _quality_stage.QUALITY_AI_OUTLIER_FLAG_COLUMN
DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD = (
    _quality_stage.DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD
)
QUALITY_IMPUTED_FLAG_COLUMN = _quality_stage.QUALITY_IMPUTED_FLAG_COLUMN
QUALITY_IMPUTED_FLAG_SUFFIX = _quality_stage.QUALITY_IMPUTED_FLAG_SUFFIX
QUALITY_INVALID_FLAG_COLUMN = _quality_stage.QUALITY_INVALID_FLAG_COLUMN
ai_anomaly_score_column = _quality_stage.ai_anomaly_score_column
ai_confidence_column = _quality_stage.ai_confidence_column
invalid_flag_column = _quality_stage.invalid_flag_column
prepare_missing_outliers_handled_features = (
    _quality_stage.prepare_missing_outliers_handled_features
)
prepare_quality_features = _quality_stage.prepare_quality_features
quality_ai_outlier_flag_column = _quality_stage.quality_ai_outlier_flag_column
quality_imputed_flag_column = _quality_stage.quality_imputed_flag_column

_growth_stage = import_module("geas35.preprocessing.growth_stage_features")
GROWTH_STAGE_DAT_COLUMN = _growth_stage.GROWTH_STAGE_DAT_COLUMN
GROWTH_STAGE_NAME_COLUMN = _growth_stage.GROWTH_STAGE_NAME_COLUMN
GROWTH_STAGE_ORDER_COLUMN = _growth_stage.GROWTH_STAGE_ORDER_COLUMN
add_growth_stage_columns = _growth_stage.add_growth_stage_columns
growth_stage_rules_sha256 = _growth_stage.growth_stage_rules_sha256

__all__ = [
    "ACTION_COLUMNS",
    "ACTION_RESTORED_FLAG_SUFFIX",
    "AI_ANOMALY_SCORE_SUFFIX",
    "AI_CONFIDENCE_SUFFIX",
    "AI_OUTLIER_FLAG_SUFFIX",
    "BINARY_ACTION_COLUMNS",
    "DEFAULT_AGGREGATE_FLAG",
    "DEFAULT_AI_OUTLIER_FLAG_COLUMN",
    "DEFAULT_AI_WARNING_FLAG_COLUMN",
    "DEFAULT_DOMAIN_RANGE_PATH",
    "DEFAULT_HIGH_CONFIDENCE_FLAG_COLUMN",
    "DEFAULT_IMPUTATION_CONFIDENCE_THRESHOLD",
    "DEFAULT_IMPUTED_FLAG_COLUMN",
    "DEFAULT_RULE_AGGREGATE_FLAG",
    "DEFAULT_RULE_FLAG_PREFIX",
    "DEFAULT_TCN_FLAG_COLUMN",
    "DEFAULT_TCN_PRED_VALUE_COLUMN",
    "DEFAULT_TCN_SCORE_COLUMN",
    "DEFAULT_TCN_WARNING_FLAG_COLUMN",
    "DERIVED_RENAME_MAP",
    "DomainRangeRule",
    "EXTERNAL_STATE_COLUMNS",
    "FEATURE_COLUMNS",
    "HUMIDITY_PERCENT_COLUMNS",
    "GROWTH_STAGE_DAT_COLUMN",
    "GROWTH_STAGE_NAME_COLUMN",
    "GROWTH_STAGE_ORDER_COLUMN",
    "INVALID_FLAG_SUFFIX",
    "INTERNAL_STATE_COLUMNS",
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
    "SEGMENT_COLUMN",
    "TCNOutlierConfig",
    "action_restored_flag_column",
    "ai_anomaly_score_column",
    "ai_confidence_column",
    "ai_outlier_flag_column",
    "ai_warning_flag_column",
    "add_growth_stage_columns",
    "add_missing_flags",
    "apply_controller_value_policy",
    "apply_default_domain_flags",
    "apply_default_rule_based_outlier_flags",
    "apply_domain_range_flags",
    "canonicalize_feature_units",
    "canonicalize_for_derived",
    "coerce_geas_numeric_features",
    "coerce_numeric_columns",
    "controller_value_column",
    "ensure_columns",
    "ensure_geas_feature_columns",
    "apply_outlier_flags",
    "apply_rule_based_outlier_flags",
    "apply_smartfarm_korea_domain_flags",
    "apply_tcn_outlier_flags",
    "high_confidence_outlier_flag_column",
    "growth_stage_rules_sha256",
    "invalid_flag_column",
    "STATE_COLUMNS",
    "SYSTEM_STATE_COLUMNS",
    "TIME_COLUMN",
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
    "outlier_imputed_flag_column",
    "quality_ai_outlier_flag_column",
    "quality_imputed_flag_column",
    "raw_value_column",
    "restore_action_columns_from_control_log",
    "rule_outlier_flag_column",
    "standardize_geas_input_schema",
    "tcn_outlier_flag_column",
    "tcn_pred_value_column",
    "tcn_score_column",
    "tcn_warning_flag_column",
]
