"""
=====================================================
🚀 ExoHabitAI — Model Evaluation Module
Production-ready classification evaluation
=====================================================
"""

from typing import Dict, Any
import logging
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------
# LOGGER
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------------------------------

def evaluate_classification_model(model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate a classification model.

    Returns:
    --------
    dict containing:
        accuracy
        precision
        recall
        f1_score
        roc_auc
        confusion_matrix
        class_distribution
        prediction_distribution
    """

    logger.info("📊 Evaluating classification model...")

    # -------------------------------------------------
    # SAFE PREDICTIONS
    # -------------------------------------------------

    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {}

    # -------------------------------------------------
    # BASIC METRICS
    # -------------------------------------------------

    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_test, y_pred, zero_division=0))

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # -------------------------------------------------
    # CLASS DISTRIBUTION (VERY IMPORTANT FOR SCIENCE)
    # -------------------------------------------------

    metrics["class_distribution"] = {
        "true_0": int(np.sum(y_test == 0)),
        "true_1": int(np.sum(y_test == 1)),
    }

    metrics["prediction_distribution"] = {
        "pred_0": int(np.sum(y_pred == 0)),
        "pred_1": int(np.sum(y_pred == 1)),
    }

    # -------------------------------------------------
    # ROC-AUC (SAFE VERSION)
    # -------------------------------------------------

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        else:
            metrics["roc_auc"] = None
    except Exception as e:
        logger.warning(f"⚠️ ROC-AUC calculation failed: {e}")
        metrics["roc_auc"] = None

    # -------------------------------------------------
    # FULL CLASSIFICATION REPORT
    # -------------------------------------------------

    try:
        metrics["classification_report"] = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        )
    except Exception:
        metrics["classification_report"] = {}

    logger.info("✅ Model evaluation completed")

    return metrics