"""
prediction_service.py

Purpose:
    Ties every pipeline stage together into one decision, exactly as
    described in the project brief:

        UPLOAD -> FILE VALIDATION -> QUALITY CHECK -> PLANT CHECK
               -> CNN CLASSIFICATION -> CONFIDENCE/MARGIN/ENTROPY/
                  CONSISTENCY CHECK -> DECISION -> RESULT

    The guiding rule: if there is not enough evidence, say so. Never
    guess just to produce an answer.

Week:
    AI/ML Foundation - Week 8
"""

import logging

import numpy as np

import config
import image_validator
import quality_checker
import plant_detector
import ood_detector
from model_loader import get_model, ModelUnavailableError

log = logging.getLogger("week8")


class InvalidUploadError(ValueError):
    """The uploaded file itself is invalid (wrong type, corrupted, etc.)."""


# ---------------------------------------------------------
# RESULT STATUSES
# ---------------------------------------------------------

STATUS_CLEAR = "clear"
STATUS_UNCERTAIN = "uncertain"
STATUS_POOR_QUALITY = "poor_quality"
STATUS_NON_PLANT = "non_plant"
STATUS_OUT_OF_DOMAIN = "out_of_domain"


def _format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


def _top_k(probabilities, k):
    order = np.argsort(probabilities)[::-1][:k]
    return [
        {
            "class_name": config.CLASS_NAMES[i],
            "label": _format_class_name(config.CLASS_NAMES[i]),
            "confidence": round(float(probabilities[i]) * 100, 2),
        }
        for i in order
    ]


def analyze_upload(filename, file_bytes):
    """
    Run the full pipeline on one uploaded file.

    Returns a result dict (see the STATUS_* constants above for the
    possible "status" values). Raises InvalidUploadError for a bad
    file, and lets ModelUnavailableError propagate if the model
    cannot be loaded - both are handled by app.py.
    """

    # -------------------------------------------------
    # 1. FILE VALIDATION
    # -------------------------------------------------

    file_result = image_validator.validate_file(filename, file_bytes)
    if not file_result.ok:
        raise InvalidUploadError(file_result.reason)

    image = file_result.image

    # -------------------------------------------------
    # 2. IMAGE QUALITY CHECK
    # -------------------------------------------------

    quality_result = quality_checker.check_quality(image)
    if not quality_result.ok:
        return {
            "status": STATUS_POOR_QUALITY,
            "message": quality_result.reason,
            "prediction": None,
            "top3": None,
            "evidence": {"quality": quality_result.metrics},
        }

    # -------------------------------------------------
    # 3. PLANT / NON-PLANT CHECK
    # (independent of the CNN - see plant_detector.py)
    # -------------------------------------------------

    plant_result = plant_detector.check_plant_presence(image)
    if not plant_result.ok:
        return {
            "status": STATUS_NON_PLANT,
            "message": plant_result.reason,
            "prediction": None,
            "top3": None,
            "evidence": {
                "quality": quality_result.metrics,
                "vegetation_ratio": plant_result.vegetation_ratio,
            },
        }

    # -------------------------------------------------
    # 4. CNN CLASSIFICATION (with TTA)
    # -------------------------------------------------

    model = get_model()  # raises ModelUnavailableError if missing

    rgb_uint8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    ood_result = ood_detector.analyze(model, rgb_uint8)

    evidence = {
        "quality": quality_result.metrics,
        "vegetation_ratio": plant_result.vegetation_ratio,
        "top1_confidence": round(ood_result.top1_confidence * 100, 2),
        "top2_confidence": round(ood_result.top2_confidence * 100, 2),
        "margin": round(ood_result.margin * 100, 2),
        "entropy_normalized": round(ood_result.entropy_normalized, 3),
        "consistency_ratio": round(ood_result.consistency_ratio, 2),
    }

    # -------------------------------------------------
    # 5 & 6. CONFIDENCE/MARGIN CHECK, then DOMAIN/OOD CHECK
    # -------------------------------------------------

    if not ood_detector.passes_confidence_check(ood_result):
        return {
            "status": STATUS_UNCERTAIN,
            "message": (
                "The image appears to contain a plant, but the model does "
                "not have enough evidence to reliably identify the "
                "condition. Please upload a clearer close-up image of the "
                "affected leaf."
            ),
            "prediction": None,
            "top3": None,
            "evidence": evidence,
        }

    if not ood_detector.passes_domain_check(ood_result):
        return {
            "status": STATUS_OUT_OF_DOMAIN,
            "message": (
                "The uploaded image may be a valid plant photo, but it does "
                "not appear sufficiently similar to the classes this model "
                "was trained on. Try a clear close-up image of a plant leaf "
                "similar to the supported dataset (see 'Supported Classes')."
            ),
            "prediction": None,
            "top3": None,
            "evidence": evidence,
        }

    # -------------------------------------------------
    # 7. CLEAR RESULT
    # -------------------------------------------------

    top3 = _top_k(ood_result.all_probabilities, config.TOP_K_PREDICTIONS)

    return {
        "status": STATUS_CLEAR,
        "message": "Analysis complete.",
        "prediction": top3[0],
        "top3": top3,
        "evidence": evidence,
    }
