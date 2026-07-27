"""Quality model interfaces and baselines for preprocessing."""

from geas35.models.quality.base import (
    BaseQualityModel,
    QualityModelOutput,
    validate_observation_columns,
)
from geas35.models.quality.baseline import MedianQualityModel
from geas35.models.quality.evaluation import (
    AnomalyDetectionMetrics,
    ConfusionMatrix,
    SyntheticAnomalyDetectionResult,
    SyntheticAnomalyInjection,
    SyntheticMaskingResult,
    evaluate_synthetic_anomaly_detection,
    evaluate_synthetic_masking,
    make_synthetic_anomaly_injection,
    make_synthetic_mask,
)
from geas35.models.quality.hyperparameters import (
    CandidateTrainingResult,
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    GridSearchHyperparameterOptimizer,
    HyperparameterCandidate,
    HyperparameterObjective,
    HyperparameterOptimizationResult,
    HyperparameterOptimizer,
    MetricGoal,
    write_hyperparameter_artifact,
)
from geas35.models.quality.modern_tcn import (
    ModernTCNConfig,
    ModernTCNQualityModel,
    train_modern_tcn_candidate,
)
from geas35.models.quality.patch_tst import (
    PatchTSTConfig,
    PatchTSTQualityModel,
    train_patch_tst_candidate,
)
from geas35.models.quality.rule_only import RuleOnlyQualityModel
from geas35.models.quality.selection_report import (
    AnomalyDetectionPerformance,
    ConfusionMatrixReport,
    OnlineEfficiencyMetrics,
    QualityModelComparisonEntry,
    QualityModelComparisonReport,
    QualityModelEvaluationReport,
    QualityModelSelectionCandidate,
    ReconstructionPerformance,
    build_quality_model_comparison_report,
    estimate_model_size_bytes,
    evaluate_quality_model_for_report,
    measure_online_inference_efficiency,
    write_quality_model_comparison_report,
)
from geas35.models.quality.thresholds import (
    GridSearchThresholdOptimizer,
    ThresholdOptimizationResult,
    ThresholdOptimizer,
)
from geas35.models.quality.timesnet import (
    TimesNetConfig,
    TimesNetQualityModel,
    train_timesnet_candidate,
)

__all__ = [
    "AnomalyDetectionMetrics",
    "AnomalyDetectionPerformance",
    "BaseQualityModel",
    "CandidateTrainingResult",
    "ConfusionMatrix",
    "ConfusionMatrixReport",
    "EarlyStoppingConfig",
    "EarlyStoppingTracker",
    "GridSearchThresholdOptimizer",
    "GridSearchHyperparameterOptimizer",
    "HyperparameterCandidate",
    "HyperparameterObjective",
    "HyperparameterOptimizationResult",
    "HyperparameterOptimizer",
    "MedianQualityModel",
    "MetricGoal",
    "ModernTCNConfig",
    "ModernTCNQualityModel",
    "OnlineEfficiencyMetrics",
    "PatchTSTConfig",
    "PatchTSTQualityModel",
    "QualityModelComparisonEntry",
    "QualityModelComparisonReport",
    "QualityModelEvaluationReport",
    "QualityModelOutput",
    "QualityModelSelectionCandidate",
    "ReconstructionPerformance",
    "RuleOnlyQualityModel",
    "SyntheticAnomalyDetectionResult",
    "SyntheticAnomalyInjection",
    "SyntheticMaskingResult",
    "ThresholdOptimizationResult",
    "ThresholdOptimizer",
    "TimesNetConfig",
    "TimesNetQualityModel",
    "build_quality_model_comparison_report",
    "estimate_model_size_bytes",
    "evaluate_quality_model_for_report",
    "evaluate_synthetic_anomaly_detection",
    "evaluate_synthetic_masking",
    "make_synthetic_anomaly_injection",
    "make_synthetic_mask",
    "measure_online_inference_efficiency",
    "train_modern_tcn_candidate",
    "train_patch_tst_candidate",
    "train_timesnet_candidate",
    "validate_observation_columns",
    "write_hyperparameter_artifact",
    "write_quality_model_comparison_report",
]
