"""
model_loader.py

Purpose:
    Load the actual trained Week 6 CNN, once, and cache it. No fake
    model, no random-weight fallback: if the real model file cannot
    be found or loaded, this raises a clear error rather than letting
    the app silently serve made-up predictions.

Week:
    AI/ML Foundation - Week 8
"""

import logging

import config

log = logging.getLogger("week8")

_model = None
_model_error = None


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


def get_model():
    """
    Return the cached trained model, loading it on first use.

    Raises:
        ModelUnavailableError if the model file is missing, or does
        not have the expected number of output classes.
    """

    global _model, _model_error

    if _model is not None:
        return _model

    if _model_error is not None:
        # Loading already failed once this process; don't retry every
        # request (that would be slow and would still fail the same way).
        raise _model_error

    if not config.MODEL_PATH.exists():
        _model_error = ModelUnavailableError(
            f"Trained model file not found at: {config.MODEL_PATH}\n"
            "This app does not train or fake a model - it expects the "
            "real Week 6 model file to already exist there. See "
            "README.md for how to configure MODEL_PATH."
        )
        raise _model_error

    try:
        from tensorflow.keras.models import load_model

        log.info("Loading trained model from %s", config.MODEL_PATH)
        model = load_model(config.MODEL_PATH, compile=False)
    except Exception as exc:  # noqa: BLE001 - we deliberately wrap any load failure
        _model_error = ModelUnavailableError(
            f"The model file at {config.MODEL_PATH} could not be loaded: {exc}"
        )
        raise _model_error

    output_size = model.output_shape[-1]
    if output_size != config.NUM_CLASSES:
        _model_error = ModelUnavailableError(
            f"Model outputs {output_size} classes but {config.NUM_CLASSES} "
            "class names are configured. Refusing to serve predictions "
            "with a mismatched class list."
        )
        raise _model_error

    _model = model
    return _model


def model_status():
    """Lightweight status check for the /health endpoint (no load attempt)."""

    return {
        "model_path": str(config.MODEL_PATH),
        "model_file_exists": config.MODEL_PATH.exists(),
        "model_loaded": _model is not None,
    }
