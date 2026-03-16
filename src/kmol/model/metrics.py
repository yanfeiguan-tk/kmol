import json
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Callable, Optional, NamedTuple, Tuple, Iterable, Any, Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)

from kmol.core.helpers import Namespace
from kmol.core.logger import LOGGER as logging
from kmol.core.observers import EventManager


class MetricType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTICLASS = "multiclass_classification"


class MetricConfiguration(NamedTuple):
    type: MetricType
    calculator: Callable
    uses_threshold: bool = False
    maximize: bool = True


class CustomMetrics:
    @staticmethod
    def __get_ranks(array: List[float]) -> np.ndarray:
        _, ranks, counts = np.unique(array, return_inverse=True, return_counts=True)

        clone = ranks.copy()
        cumulative_sum = -1

        for index, count in enumerate(counts):
            cumulative_sum += count
            ranks[np.where(clone == index)[0]] = cumulative_sum

        return ranks

    @staticmethod
    def pearson_correlation_coefficient(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.pearsonr(ground_truth, predictions)[0]

    @staticmethod
    def spearman_correlation_coefficient(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.spearmanr(ground_truth, predictions).correlation

    @staticmethod
    def kullback_leibler_divergence(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.entropy(ground_truth, predictions)

    @staticmethod
    def jensen_shannon_divergence(ground_truth: List[float], predictions: List[float]) -> float:
        return math.exp(distance.jensenshannon(ground_truth, predictions))

    @staticmethod
    def rank_quality(ground_truth: List[float], predictions: List[float]) -> float:
        """
        For this metric, values don't matter, only the order.
        This helps if you plan to use a regression model for ranking, the exact values are not important.
        The best values is 1, the worst is 0.
        """
        if len(ground_truth) == 1:
            return 1.0

        ground_truth = CustomMetrics.__get_ranks(ground_truth)
        predictions = CustomMetrics.__get_ranks(predictions)

        if len(ground_truth) == 2:
            return float(ground_truth[0] == predictions[0])

        samples_count = ground_truth.shape[0]
        worst_possible_outcome = np.arange(int(samples_count % 2 == 0), samples_count, step=2).sum() * 2

        return 1 - np.sum(np.abs(ground_truth - predictions)) / worst_possible_outcome

    @staticmethod
    def censored_mae(ground_truth: np.ndarray, predictions: np.ndarray, censoring: np.ndarray) -> float:
        """
        Mean Absolute Error for censored data.
        
        Only computes MAE on uncensored observations (censoring == 0).
        For censored observations, we cannot compute true error since we don't know the true value.
        
        Args:
            ground_truth: Array of shape (n,) with observed values or censoring thresholds
            predictions: Array of shape (n,) with predicted values
            censoring: Array of shape (n,) with censoring indicators:
                       0 = uncensored, -1 = left-censored, 1 = right-censored
        
        Returns:
            MAE computed only on uncensored observations
        """
        # Extract uncensored observations
        uncensored_mask = (censoring == 0)
        
        if not np.any(uncensored_mask):
            # No uncensored data, return NaN
            return np.nan
        
        uncensored_true = ground_truth[uncensored_mask]
        uncensored_pred = predictions[uncensored_mask]
        
        return mean_absolute_error(uncensored_true, uncensored_pred)

    @staticmethod
    def censored_r2(ground_truth: np.ndarray, predictions: np.ndarray, censoring: np.ndarray) -> float:
        """
        R-squared for censored data.
        
        Only computes R2 on uncensored observations (censoring == 0).
        For censored observations, we cannot compute true error since we don't know the true value.
        
        Args:
            ground_truth: Array of shape (n,) with observed values or censoring thresholds
            predictions: Array of shape (n,) with predicted values  
            censoring: Array of shape (n,) with censoring indicators:
                       0 = uncensored, -1 = left-censored, 1 = right-censored
        
        Returns:
            R2 computed only on uncensored observations
        """
        # Extract uncensored observations
        uncensored_mask = (censoring == 0)
        
        if not np.any(uncensored_mask):
            # No uncensored data, return NaN
            return np.nan
        
        uncensored_true = ground_truth[uncensored_mask]
        uncensored_pred = predictions[uncensored_mask]
        
        return r2_score(uncensored_true, uncensored_pred)


class AvailableMetrics:
    # fmt: off
    MAE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_absolute_error, maximize=False)
    MSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_squared_error, maximize=False)
    RMSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=partial(mean_squared_error, squared=False), maximize=False)  # noqa: E501
    R2 = MetricConfiguration(type=MetricType.REGRESSION, calculator=r2_score)
    PEARSON = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.pearson_correlation_coefficient)
    SPEARMAN = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.spearman_correlation_coefficient)
    KL_DIV = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.kullback_leibler_divergence, maximize=False)  # noqa: E501
    JS_DIV = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.jensen_shannon_divergence, maximize=False)
    CHEBYSHEV = MetricConfiguration(type=MetricType.REGRESSION, calculator=distance.chebyshev, maximize=False)
    MANHATTAN = MetricConfiguration(type=MetricType.REGRESSION, calculator=distance.cityblock, maximize=False)
    RANK_QUALITY = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.rank_quality)
    CENSORED_MAE = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.censored_mae, maximize=False)
    CENSORED_R2 = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.censored_r2)

    ROC_AUC = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=roc_auc_score)
    PR_AUC = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=average_precision_score)
    ACCURACY = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=accuracy_score, uses_threshold=True)
    ACCURACY_MULTICLASS = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=accuracy_score, uses_threshold=False)
    PRECISION = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=partial(precision_score, zero_division=1), uses_threshold=True)  # noqa: E501
    RECALL = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=partial(recall_score, zero_division=1), uses_threshold=True)  # noqa: E501
    F1 = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=f1_score, uses_threshold=True)
    COHEN_KAPPA = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=cohen_kappa_score, uses_threshold=True)
    JACCARD = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=jaccard_score, uses_threshold=True)

    TOP_1_ACCURACY = MetricConfiguration(type=MetricType.MULTICLASS, calculator=partial(top_k_accuracy_score, k=1))
    TOP_10_ACCURACY = MetricConfiguration(type=MetricType.MULTICLASS, calculator=partial(top_k_accuracy_score, k=10))
    TOP_100_ACCURACY = MetricConfiguration(type=MetricType.MULTICLASS, calculator=partial(top_k_accuracy_score, k=100))
    # fmt: on


class PredictionProcessor:
    def __init__(
        self,
        metrics: List[str],
        threshold: Optional[float] = None,
        error_value: Any = np.nan,
    ):
        self._metrics = self._map_metrics(metrics)
        self._threshold = threshold
        self._error_value = error_value

    def _map_metrics(self, metrics: List[str]) -> Dict[str, MetricConfiguration]:
        return {metric: getattr(AvailableMetrics, metric.upper()) for metric in metrics}

    def _is_censored_metric(self, metric_name: str) -> bool:
        """Check if a metric requires censoring information."""
        return metric_name.upper().startswith('CENSORED_')

    def _needs_predictions(self) -> bool:
        return any(metric.uses_threshold for metric in self._metrics.values())

    @classmethod
    def detach(cls, tensor: torch.Tensor) -> List[List[float]]:
        return tensor.cpu().detach().tolist()

    @classmethod
    def apply_threshold(cls, logits: torch.Tensor, threshold: float) -> np.ndarray:
        if threshold is None:
            return np.array(cls.detach(logits))

        predictions = torch.sigmoid(logits)
        predictions = cls.detach(predictions)

        return np.where(np.less(predictions, threshold), 0, 1)

    @classmethod
    def compute_statistics(
        cls,
        metrics: Namespace,
        statistics: Iterable[Callable] = (np.min, np.max, np.mean, np.median, np.std),
    ) -> Namespace:
        results = defaultdict(list)
        for name, values in vars(metrics).items():
            for statistic in statistics:
                results[name].append(statistic(values))

        return Namespace(**results)

    def _prepare(
        self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]
    ) -> Tuple[List[List[float]], List[List[float]], Optional[List[List[float]]], Optional[List[List[float]]]]:
        logits = torch.cat(logits)
        ground_truth = torch.cat(ground_truth)
        payload = Namespace(logits=logits, ground_truth=ground_truth)
        EventManager.dispatch_event(event_name="before_metric", payload=payload)

        # Check if we have censored metrics that need censoring indicators
        has_censored_metrics = any(self._is_censored_metric(name) for name in self._metrics.keys())
        
        # For censored data: ground_truth has shape (batch, 2) where
        # ground_truth[:, 0] = observed value/threshold
        # ground_truth[:, 1] = censoring indicator (0=uncensored, -1=left, 1=right)
        censoring_indicators = None
        if has_censored_metrics and payload.ground_truth.shape[1] >= 2:
            # Extract censoring indicators from second column
            censoring_indicators = payload.ground_truth[:, 1:2]
            # Keep only the first column as ground truth values
            payload.ground_truth = payload.ground_truth[:, 0:1]
        
        # Transpose tensors
        ground_truth = payload.ground_truth.t()
        logits = payload.logits.t()
        if censoring_indicators is not None:
            censoring_indicators = censoring_indicators.t()

        # move values to CPU, get predictions from logits if needed
        ground_truth = self.detach(ground_truth)
        predictions = None
        if self._needs_predictions():
            predictions = self.apply_threshold(logits, self._threshold).tolist()
        logits = self.detach(logits)
        if censoring_indicators is not None:
            censoring_indicators = self.detach(censoring_indicators)

        # remove missing labels
        mask = np.isnan(ground_truth)
        for i in range(len(ground_truth)):
            ground_truth[i] = np.delete(ground_truth[i], mask[i])
            logits[i] = np.delete(logits[i], mask[i])

            if predictions:
                predictions[i] = np.delete(predictions[i], mask[i])
            
            if censoring_indicators is not None and i < len(censoring_indicators):
                censoring_indicators[i] = np.delete(censoring_indicators[i], mask[i])

        return ground_truth, logits, predictions, censoring_indicators

    def compute_metrics(self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]) -> Namespace:
        metrics = defaultdict(list)
        if self._metrics:
            ground_truth, logits, predictions, censoring_indicators = self._prepare(ground_truth=ground_truth, logits=logits)

            for target_index in range(len(ground_truth)):
                for metric_name, metric_settings in self._metrics.items():
                    if metric_settings.type == MetricType.MULTICLASS:
                        labels = np.array(logits).T
                    elif metric_settings.uses_threshold:
                        labels = predictions[target_index]
                    else:
                        labels = logits[target_index]

                    try:
                        # Check if this is a censored metric that needs censoring indicators
                        if self._is_censored_metric(metric_name) and censoring_indicators is not None:
                            # Pass censoring indicators as third argument
                            computed_value = metric_settings.calculator(
                                ground_truth[target_index], 
                                labels,
                                censoring_indicators[target_index]
                            )
                        else:
                            computed_value = metric_settings.calculator(ground_truth[target_index], labels)
                        
                        if not metric_settings.maximize:
                            computed_value *= -1
                    except (ValueError, IndexError) as e:
                        computed_value = self._error_value

                    metrics[metric_name].append(computed_value)

        return Namespace(**metrics)

    def find_best_threshold(self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]) -> List[float]:
        logits = [torch.sigmoid(tensor) for tensor in logits]
        ground_truth, logits, _ = self._prepare(ground_truth=ground_truth, logits=logits)

        best = []
        for i in range(len(ground_truth)):
            false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth[i], logits[i])
            best.append(thresholds[np.argmax(true_positive_rate - false_positive_rate)])

        return best


class AbstractMetricLogger(metaclass=ABCMeta):
    @abstractmethod
    def log_header(self, headers: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_content(self, content: Namespace) -> None:
        raise NotImplementedError


class JsonLogger(AbstractMetricLogger):
    def log_header(self, headers: List[str]) -> None:
        logging.info("--------------------------------------------------------------")
        logging.info(headers)
        logging.info("--------------------------------------------------------------")

    def log_content(self, content: Namespace) -> None:
        logging.info(json.dumps(vars(content)))


class CsvLogger(AbstractMetricLogger):
    def log_header(self, headers: List[str]) -> None:
        logging.info("--------------------------------------------------------------")
        logging.info("metric,{}".format(",".join(headers)))
        logging.info("--------------------------------------------------------------")

    def log_content(self, content: Namespace) -> None:
        for name, values in vars(content).items():
            values = [str(value) for value in values]
            logging.info("{},{}".format(name, ",".join(values)))
