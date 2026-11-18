"""PdM-CMaps: predictive maintenance for NASA C-MAPSS turbofans (RUL, anomaly detection, monitoring layer)."""

from pdm_cmaps.analysis.stats import ThesisHypothesisAnalyzer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.detection.spc import EnvelopeAnomalyDetector, StatisticalAnomalyDetector
from pdm_cmaps.detection.virtual_sensor import VirtualSensor
from pdm_cmaps.domain.rul import compute_piecewise_rul, evaluate_rul_predictions, nasa_scoring_function
from pdm_cmaps.modeling.benchmark import RULBenchmark
from pdm_cmaps.modeling.registry import get_model, get_model_pool
from pdm_cmaps.pipeline.config import PipelineConfig
from pdm_cmaps.pipeline.pipeline import PdMPipeline

__all__ = [
    "CMapssLoader", "compute_piecewise_rul", "evaluate_rul_predictions", "nasa_scoring_function",
    "RegimeDecoupler", "StatisticalAnomalyDetector", "EnvelopeAnomalyDetector", "VirtualSensor",
    "RULBenchmark", "get_model", "get_model_pool", "ThesisHypothesisAnalyzer", "PdMPipeline", "PipelineConfig",
]
