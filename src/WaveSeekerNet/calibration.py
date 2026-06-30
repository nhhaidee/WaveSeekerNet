"""
WaveSeekerNet calibration module.

Contains the isotonic calibration wrapper (:class:`IsotonicCalibrator`)
and functions to compute overall and class-wise calibration metrics.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)


class IsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """Isotonic regression calibrator for multi-class probabilities (One-vs-Rest).

    Wraps scikit-learn's IsotonicRegression to calibrate multi-class
    probabilities by fitting a binary calibrator for each class and
    re-normalizing the outputs.

    Parameters
    ----------
    n_classes : int, default=3
        The number of classes to calibrate.
    """

    def __init__(self, n_classes: int = 3) -> None:
        self.n_classes = n_classes
        self.regressors: list[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        """Fit the isotonic calibrators on the predicted probabilities.

        Parameters
        ----------
        probs : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated predicted probabilities.
        y : np.ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        IsotonicCalibrator
            Fitted calibrator instance.
        """
        self.regressors = []
        for i in range(self.n_classes):
            # One-vs-Rest binary target
            y_binary = (y == i).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[:, i], y_binary)
            self.regressors.append(iso)
        
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.

        Parameters
        ----------
        probs : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated predicted probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Calibrated predicted probabilities.
        """
        check_is_fitted(self, attributes=["classes_"])
        
        calibrated = np.zeros_like(probs)
        for i, iso in enumerate(self.regressors):
            calibrated[:, i] = iso.predict(probs[:, i])
            
        # Re-normalize to ensure they sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        return calibrated / row_sums

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Predict class labels after calibration.

        Parameters
        ----------
        probs : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated predicted probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return np.argmax(self.predict_proba(probs), axis=1)


def compute_calibration_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    n_classes: int = 3,
) -> tuple[float, float, float, float, list[dict[str, Any]]]:
    """Compute overall calibration metrics using fixed and adaptive binning.

    Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Adaptive Calibration Error (ACE), and Mean Calibration Signed Gap (MCS).

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities.
    y_true : np.ndarray of shape (n_samples,)
        True class labels.
    n_bins : int, default=10
        Number of bins for calibration error estimation.
    n_classes : int, default=3
        Number of classes.

    Returns
    -------
    ece : float
        Expected Calibration Error.
    mce : float
        Maximum Calibration Error.
    ace : float
        Adaptive Calibration Error.
    mcs : float
        Mean Calibration Signed Gap (direction of miscalibration).
    bin_stats : list of dict
        Statistics for each bin, including boundaries, counts, accuracies,
        confidences, gaps, and class distribution.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    N = len(y_true)
    fixed_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    ece, mce, mcs = 0.0, 0.0, 0.0
    bin_stats = [] 
    for i, (bin_lower, bin_upper) in enumerate(zip(fixed_boundaries[:-1], fixed_boundaries[1:])):
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        count_in_bin = np.sum(in_bin)
        
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            gap = float(np.abs(conf_in_bin - acc_in_bin))
            signed_gap = float(conf_in_bin - acc_in_bin)
            
            ece += gap * prop_in_bin
            mcs += signed_gap * prop_in_bin
            mce = max(mce, gap)
            
            bin_y_true = y_true[in_bin]
            class_counts = [int(np.sum(bin_y_true == c)) for c in range(n_classes)]
            bin_stats.append({
                "lower": bin_lower,
                "upper": bin_upper,
                "count": int(count_in_bin),
                "prop": float(prop_in_bin),
                "acc": float(acc_in_bin),
                "conf": float(conf_in_bin),
                "gap": gap,
                "classes": class_counts
            })
        else:
            bin_stats.append({
                "lower": bin_lower,
                "upper": bin_upper,
                "count": 0,
                "prop": 0.0,
                "acc": None,
                "conf": None,
                "gap": None,
                "classes": [0] * n_classes
            })
    
    sorted_indices = np.argsort(confidences)
    sorted_confs = confidences[sorted_indices]
    sorted_accs = accuracies[sorted_indices]
    
    ace = 0.0
    splits = np.array_split(np.arange(N), n_bins)
    for chunk in splits:
        if len(chunk) == 0:
            continue
        ace += np.abs(np.mean(sorted_confs[chunk]) - np.mean(sorted_accs[chunk]))
    ace /= n_bins
    
    return ece, mce, ace, mcs, bin_stats


def compute_class_wise_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    n_classes: int = 3,
) -> dict[str, Any]:
    """Compute class-wise calibration metrics (ECE, MCE, MCS, ACE) for each class.

    Calculates metrics using One-vs-Rest binary evaluation per class.

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities.
    y_true : np.ndarray of shape (n_samples,)
        True class labels.
    n_bins : int, default=10
        Number of bins for calibration error estimation.
    n_classes : int, default=3
        Number of classes.

    Returns
    -------
    dict
        A dictionary containing overall macro averages and per-class lists for
        ECE, MCE, ACE, and MCS.
    """
    N = len(y_true)
    fixed_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    class_ece = []
    class_mce = []
    class_mcs = []
    class_ace = []

    for c in range(n_classes):
        c_probs = probs[:, c]
        c_trues = (y_true == c).astype(float)

        # --- Fixed Binning Metrics (ECE, MCE, MCS) ---
        c_ece_accum = 0.0
        c_mce_val = 0.0
        c_mcs_accum = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = fixed_boundaries[i], fixed_boundaries[i + 1]
            if i == 0:
                in_bin = (c_probs >= bin_lower) & (c_probs <= bin_upper)
            else:
                in_bin = (c_probs > bin_lower) & (c_probs <= bin_upper)

            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(c_trues[in_bin])
                conf_in_bin = np.mean(c_probs[in_bin])

                gap = np.abs(conf_in_bin - acc_in_bin)
                signed_gap = conf_in_bin - acc_in_bin

                c_ece_accum += gap * prop_in_bin
                c_mcs_accum += signed_gap * prop_in_bin
                c_mce_val = max(c_mce_val, gap)

        # --- Adaptive Binning Metric (ACE) ---
        sorted_indices = np.argsort(c_probs)
        sorted_c_confs = c_probs[sorted_indices]
        sorted_c_trues = c_trues[sorted_indices]

        c_ace_accum = 0.0
        splits = np.array_split(np.arange(N), n_bins)
        for chunk in splits:
            if len(chunk) > 0:
                c_ace_accum += np.abs(np.mean(sorted_c_confs[chunk]) - np.mean(sorted_c_trues[chunk]))

        class_ece.append(c_ece_accum)
        class_mce.append(c_mce_val)
        class_mcs.append(c_mcs_accum)
        class_ace.append(c_ace_accum / n_bins)

    metrics = {
        "avg_class_ece": float(np.mean(class_ece)),
        "avg_class_mce": float(np.mean(class_mce)),
        "avg_class_ace": float(np.mean(class_ace)),
        "avg_class_mcs": float(np.mean(class_mcs)),
        "per_class_details": {
            "ece": class_ece,
            "mce": class_mce,
            "ace": class_ace,
            "mcs": class_mcs
        }
    }

    return metrics
