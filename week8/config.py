"""
config.py

Week 8 - AI/ML Foundation
All tunable thresholds and constants for the validation pipeline live
here, in one place, with an explanation of what each one does and why
its default value was chosen. Every threshold below is a heuristic
judgment call, not a value derived from a formal calibration study -
they are deliberately easy to find and change.

Nothing in this file is a secret or credential.
"""

import os
from pathlib import Path

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

WEEK8_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEEK8_DIR.parent

# The model trained and checkpointed in Week 6 (best model by validation
# loss), the same model Week 7 used for its per-class evaluation.
# Override with the MODEL_PATH environment variable if you keep the
# model somewhere else.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        REPO_ROOT / "week6" / "models" / "plant_disease_cnn_week6.keras",
    )
)

# ---------------------------------------------------------
# MODEL / PREPROCESSING
# (must match week6/src/data_loader.py exactly)
# ---------------------------------------------------------

IMAGE_SIZE = (128, 128)

# The 38 PlantVillage classes, in the exact sorted order Weeks 4-7 used
# (sorted(set(labels)) over the folder names). Index i of the model's
# softmax output corresponds to CLASS_NAMES[i]. Copied verbatim from
# week6/reports/week6_report.txt - not invented.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------
# UPLOAD / FILE VALIDATION
# ---------------------------------------------------------

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Pillow's own name for the format after decoding - cross-checked
# against the extension so a renamed file can't sneak past.
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}

MAX_UPLOAD_MB = 5

# ---------------------------------------------------------
# IMAGE QUALITY THRESHOLDS
# ---------------------------------------------------------

# Below this, the image is too small to trust - the model resizes
# everything to 128x128 anyway, so anything much smaller than that is
# being heavily upscaled and loses real detail.
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

# Variance of the Laplacian of the grayscale image (a standard, simple
# blur-detection score - sharp images have a lot of high-frequency
# edge energy, blurry ones have very little). Computed on a version of
# the image resized so its longer side is 400px, so the score is
# comparable across different upload resolutions.
# Calibrated empirically (see tests/) against a sharp synthetic image
# (~vals in the thousands) and a heavily blurred one (~vals below 5).
BLUR_THRESHOLD = 15.0

# Mean grayscale brightness, 0-255 scale.
BRIGHTNESS_MIN = 25.0    # below this: image is judged too dark
BRIGHTNESS_MAX = 232.0   # above this: image is judged blown-out/too bright

# Standard deviation of grayscale brightness. A very low value means a
# flat, low-detail image (e.g. a blank wall, a solid color).
CONTRAST_MIN = 10.0

# ---------------------------------------------------------
# PLANT / NON-PLANT (VEGETATION) HEURISTIC
# ---------------------------------------------------------
# See plant_detector.py for the full explanation and limitations.
#
# The mask is defined directly on RGB channel relationships rather
# than HSV hue, because hue becomes numerically unstable (noisy) for
# desaturated/grayish pixels - exactly the kind of pixel a photo of a
# car, laptop or skin tone is full of. Requiring the green channel to
# clearly *dominate* the other channels (by a margin, on the 0-255
# scale) is a stronger and more specific signal.
#
# "Green" pixels:  G is at least GREEN_MARGIN above both R and B
#                   (healthy foliage).
# "Yellow/brown" pixels: both R and G are at least YELLOW_MARGIN above
#                   B, and R and G are close to each other
#                   (within YELLOW_RG_MAX_DIFF) - covers chlorotic,
#                   senescent or diseased leaf tones without also
#                   matching typical skin tones (where R is usually
#                   well above G).
GREEN_MARGIN = 20
YELLOW_MARGIN = 28
YELLOW_RG_MAX_DIFF = 16

# Minimum fraction of pixels that must fall in the vegetation-colored
# range for the image to be treated as "plausibly a plant/leaf".
# Calibrated empirically (see tests/) so that synthetic car/laptop/
# skin-tone images stay below ~0.16 and synthetic healthy/diseased
# leaf images stay above ~0.4 with a safety margin either side.
VEGETATION_MIN_RATIO = 0.15

# ---------------------------------------------------------
# MODEL CONFIDENCE / DOMAIN (OUT-OF-DISTRIBUTION) THRESHOLDS
# ---------------------------------------------------------

# Minimum top-1 softmax probability to even consider a "clear" result.
MIN_ACCEPTED_CONFIDENCE = 0.55

# Minimum gap between the top-1 and top-2 softmax probabilities. A
# small gap means the model is genuinely torn between two classes.
MIN_PROBABILITY_MARGIN = 0.15

# Test-time augmentation (TTA) consistency: the image is classified
# 4 times (original, horizontal flip, and two brightness variants).
# This fraction of those runs must agree with the original prediction.
MIN_CONSISTENCY_RATIO = 0.75

# Normalized entropy of the softmax distribution (entropy divided by
# log(NUM_CLASSES), so it always falls in [0, 1] regardless of the
# number of classes). Higher means the model's probability mass is
# spread out over many classes rather than concentrated - a sign the
# model does not really "recognize" the image.
MAX_NORMALIZED_ENTROPY = 0.55

# ---------------------------------------------------------
# MISC
# ---------------------------------------------------------

TOP_K_PREDICTIONS = 3
