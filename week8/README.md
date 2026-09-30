# Week 8 — Realistic Plant Disease Image Analysis (Rebuilt)

## Project overview

This is Week 8 of an AI/ML Foundation Program: a Flask web application that
serves the plant-disease CNN trained in Weeks 4-6 (and evaluated per-class
in Week 7). Instead of the usual "upload a photo, get a disease name" demo,
this version runs the upload through a multi-stage validation pipeline
first, so the app only ever names a disease when there is real evidence for
it - and clearly says so, instead of guessing, whenever there isn't.

## Week 8 objective

The original brief for this week was "Mobile-friendly deployment - Deploy
simple web app." This rebuild keeps that goal but adds the requirement that
the deployment be **realistic and safe**: it must not force every upload
into one of the 38 known disease classes, must not treat raw softmax
confidence as real-world certainty, and must fail safely and honestly when
an image is unsuitable, low-quality, not a plant, or simply outside what the
model was trained on.

## Why this was rebuilt

The original Week 8 app ran a single blind step: **image → CNN → softmax →
prediction**, and always displayed whatever class won, however unrelated the
image was. A softmax layer always sums to 100% across its 38 known classes —
even for a photo of a car — so a high number there never proved the model
actually recognized anything. This rebuild adds a validation pipeline in
front of the CNN so the app only shows a disease name when there is real
evidence for it, and otherwise says so honestly.

## Architecture

```text
UPLOAD IMAGE
      ↓
1. FILE VALIDATION        (image_validator.py)
      ↓
2. IMAGE QUALITY CHECK    (quality_checker.py)
      ↓
3. PLANT / NON-PLANT CHECK (plant_detector.py)  — independent of the CNN
      ↓
4. CNN CLASSIFICATION + TTA (model_loader.py, ood_detector.py)
      ↓
5. CONFIDENCE + MARGIN CHECK, then DOMAIN/CONSISTENCY CHECK (ood_detector.py)
      ↓
6. DECISION                (prediction_service.py)
      ↓
FINAL RESULT: clear / uncertain / poor_quality / non_plant / out_of_domain
```

`app.py` is a thin Flask layer that calls `prediction_service.analyze_upload()`
and returns its result as JSON; `templates/index.html` + `static/` render the
five result states.

## The 5 result states

| status | Meaning | Shown to user |
|---|---|---|
| `clear` | Passed every check | 🌿 Plant detected + predicted class + confidence + top-3 + evidence |
| `uncertain` | Plant-colored, but the model's own confidence/margin is too low | "The image appears to contain a plant, but the model does not have enough evidence..." |
| `poor_quality` | Failed the quality check (blur/dark/bright/flat/too small) | The specific reason (e.g. "Image appears too blurry...") |
| `non_plant` | Failed the vegetation-color check | "No plant detected..." |
| `out_of_domain` | Plant-colored and confident, but TTA-consistency/entropy signals disagree | "may be a valid plant photo, but does not appear sufficiently similar..." |

**No disease name, confidence bar, or top-3 list is ever shown for anything
except `clear`.** This was verified directly (see Testing).

## Why the old system produced false predictions

The 38-class CNN has no "none of the above" option — softmax always
redistributes 100% of its probability mass across the 38 classes it knows,
no matter what it's shown. The old app treated the winning class and its
raw probability as the answer. This rebuild never uses the CNN's own output
to decide "is this a plant" (that would be circular), and never treats a
high softmax number, by itself, as proof of anything.

## How non-plant images are rejected (Stage 3)

`plant_detector.py` uses a lightweight, **heuristic** color rule, not object
detection and not a pretrained vision model (the brief's "Option C" —
practical and CPU-only for a student laptop, no extra multi-hundred-MB
download). A pixel counts as vegetation-colored if:

- **green dominant**: green channel is clearly above both red and blue, or
- **yellow/brown dominant**: red and green are both clearly above blue and
  close to each other (covers chlorotic/senescent/diseased leaf tones,
  without also matching typical skin tones, where red is usually well above
  green)

If fewer than `VEGETATION_MIN_RATIO` (15%, in `config.py`) of pixels match,
the image is rejected as `non_plant` **before the CNN ever runs**.

**Honest limitation:** this is a color heuristic, not a leaf detector. It was
calibrated and tested with synthetic color-patch images representing cars,
laptops, skin tones, and leaves (see `tests/test_plant_detector.py`), not
real photographs. In that testing, a green car, a green wall, or green
fabric could still pass; a fully dead/brown leaf could still be rejected. It
is one gate in the pipeline, not a claim of general object recognition, and
every threshold is in `config.py` for easy re-tuning against real photos.

## How blurry / low-quality images are handled (Stage 2)

`quality_checker.py` checks, before anything else:

- **Dimensions** — width/height below 64×64px → rejected (heavy upscaling
  loses real detail)
- **Blur** — variance of the Laplacian (a standard edge-energy blur score,
  implemented directly with NumPy, no OpenCV/SciPy dependency) below 15.0
- **Brightness** — mean grayscale value below 25 (too dark) or above 232
  (blown out)
- **Contrast** — standard deviation of grayscale below 10 (flat/blank image)

All four thresholds are named constants in `config.py`, calibrated against
synthetic sharp/blurred and dark/bright test images (see
`tests/test_quality_checker.py`), and documented there with the reasoning
behind each value.

## How unsupported / uncertain images are handled (Stages 5-6)

Once an image passes the plant check, the CNN runs on the original image
**plus 3 test-time-augmented (TTA) variants** — horizontally flipped, and
two brightness adjustments. From this, `ood_detector.py` computes:

- **top-1 confidence** and **margin** to the runner-up class
- **normalized entropy** of the full 38-class distribution
- **TTA consistency** — how often the 4 variants agree on the same class

A result is only shown as `clear` if confidence ≥ 55%, margin ≥ 15 points,
TTA agreement ≥ 75%, **and** normalized entropy ≤ 0.55. Otherwise it is
`uncertain` (confidence/margin too low) or `out_of_domain` (confident but
inconsistent/scattered).

**Honest limitation found during testing:** this particular trained model is
frequently *very* overconfident — in my synthetic tests, even plant-colored
input that didn't match any real disease pattern often still received
~100% confidence with perfect TTA agreement, so the margin/entropy/
consistency checks rarely fired on their own. I confirmed the decision logic
itself is correct (unit-tested directly, and it does correctly catch a
genuinely divided/ambiguous image as `uncertain` — see Testing), but in
practice **the vegetation-color gate (Stage 3) is doing most of the real
rejection work against non-plant images**, not the OOD layer. This is a
known, documented behavior of un-calibrated softmax classifiers, not a bug,
and it is why Stage 3 exists as an independent check rather than relying on
the CNN's confidence alone.

## How confidence is calculated and interpreted

Confidence shown to the user is the raw softmax probability of the top
class, labeled **"Model confidence"**, with an explicit note in the UI:
*"Confidence reflects the model's output among its supported classes. It is
not a guarantee that the diagnosis is correct."* Numbers are never
manufactured, rounded up, or artificially reduced — every value shown is a
real number computed from the model's output, or the prediction is withheld
entirely.

## Disease severity

This model is a classifier, not a severity estimator. The UI states plainly:
*"This model is not designed to quantify disease severity."* No severity
score is computed or implied anywhere in the app.

## Model used and preprocessing

- Model: `week6/models/plant_disease_cnn_week6.keras` (the improved,
  checkpointed model Week 7 also evaluated). **Not retrained.**
- Preprocessing matches `week6/src/data_loader.py` exactly: RGB → resize to
  128×128 with `tf.image.resize` → float32 → divide by 255.
- Class order: the 38 sorted PlantVillage class names, copied verbatim from
  `week6/reports/week6_report.txt` into `config.py`. The model loader
  verifies the model's output layer has exactly 38 units before serving any
  prediction, and refuses to run if it doesn't match.
- If the model file is missing, the app does **not** fall back to a fake
  model — `/predict` returns a 503 with a clear message, and `/health`
  reports `model_file_exists: false`.

## Supported classes

All 38 PlantVillage classes are listed in a collapsible "Supported classes"
section on the page itself, generated directly from `config.CLASS_NAMES` (no
separate list to keep in sync). The page states: *"This model was trained on
a fixed set of PlantVillage classes. Images outside this domain may be
rejected or marked uncertain."*

## Folder structure

```text
week8/
├── app.py                    Flask routes
├── config.py                 All thresholds/constants, documented
├── model_loader.py           Loads & caches the real Week 6 model
├── image_validator.py        Stage 1: file/format validation
├── quality_checker.py        Stage 2: blur/brightness/contrast/size
├── plant_detector.py         Stage 3: vegetation-color heuristic
├── ood_detector.py           Stages 4-5: TTA, confidence, margin, entropy
├── prediction_service.py     Orchestrates all stages -> final decision
├── requirements.txt
├── .gitignore
├── pytest.ini
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── tests/
    ├── test_image_validator.py
    ├── test_quality_checker.py
    ├── test_plant_detector.py
    └── test_prediction_service.py   (full pipeline, needs the real model)
```

## Security / upload handling

Uploads are processed **entirely in memory** (`io.BytesIO`) — nothing is
ever written to disk. This sidesteps upload-directory security concerns
directly: no filename to sanitize for storage, no temp file path to
construct, no temp file to clean up, no risk of arbitrary file execution.
Additional safeguards: allowed-extension check, decoded-format check
(catches a renamed file), 5 MB request size cap (`MAX_CONTENT_LENGTH`,
returns HTTP 413), and no internal exceptions or stack traces are ever
returned to the client (all exception handlers return a fixed, friendly
message and log the real error server-side).

## How to run locally

```powershell
cd week8
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The Week 6 model file
(`../week6/models/plant_disease_cnn_week6.keras`) must exist relative to this
folder, or set the `MODEL_PATH` environment variable to point elsewhere.

`requirements.txt` pins `tensorflow-cpu==2.20.0` / `keras==3.12.4` (matching
the version the model was saved with) and was tested on **Python 3.12**.
TensorFlow 2.20 does not yet publish wheels for Python 3.13 at the time of
writing — if `pip install` fails on Python 3.13, install Python 3.12
alongside it and create the virtual environment with that version instead
(`py -3.12 -m venv .venv` on Windows).

## How to deploy on Render

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Python 3 |
| Root Directory | `week8` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120` |
| Health Check Path | `/health` |
| Instance type | **At least 1–2 GB RAM** (see below) |
| Env var `PYTHON_VERSION` | `3.12.3` |

**Memory:** measured locally, one Gunicorn worker with the model loaded uses
roughly **520–570 MB**. Render's free/512 MB tier will likely run out of
memory — use a plan with more headroom, and keep `--workers 1`.

## Known limitations (read before presenting this project)

- This is a **student AI/ML project demonstration**, not a certified
  diagnostic tool, and not a substitute for an agricultural expert.
- The CNN only knows 38 PlantVillage classes; it cannot identify any disease
  or plant outside that set, and the app cannot guarantee it will always
  correctly say so (see below).
- The plant/non-plant check is a **color heuristic**, calibrated against
  synthetic test images, not real photographs — it can be fooled by
  strongly green/yellow-brown non-plant objects, and can reject a real but
  unusually colored leaf.
- The OOD/domain check (margin, entropy, TTA consistency) is real and
  unit-tested, but in practice this specific model is often overconfident
  even on ambiguous input, so it fires less often than its four thresholds
  might suggest — the vegetation-color gate is the primary practical
  safeguard, not this layer.
- Reported accuracy figures (Week 6: ~92.9% test accuracy; per-class figures
  in Week 7) describe performance **on the PlantVillage test set only**.
  Real phone photos (different backgrounds, lighting, multiple leaves) have
  not been measured and should be expected to score lower.
- No accounts, database, or persistent storage. Nothing uploaded is saved.

## Testing

### Automated (`pytest`, from inside `week8/`)

```powershell
pip install pytest
pytest -v
```

27 tests across 4 files. `test_prediction_service.py` needs the real Week 6
model file to be present and is automatically **skipped** (not failed) if
it isn't found, so the rest of the suite still runs without it.

| File | Covers |
|---|---|
| `test_image_validator.py` | missing/empty file, wrong extension, corrupted/truncated file, renamed-format file, valid JPEG |
| `test_quality_checker.py` | too small, too dark, too bright, low-contrast, blurry, and a passing "good" image; blur-score ordering |
| `test_plant_detector.py` | synthetic car/laptop/skin/sky vs. healthy/yellow-green/diseased-leaf colors, above/below the threshold |
| `test_prediction_service.py` | full pipeline: supported leaf → `clear`; car/laptop/skin → `non_plant`; blurry/dark/tiny → `poor_quality`; a genuinely split/ambiguous image → `uncertain`/`out_of_domain`; rejected results never carry a top-3 list |

**Note on the test images:** none of these are real photographs (no dataset
is bundled). They are synthetic color patches built specifically to isolate
one signal at a time (e.g. "mostly gray-blue, low green" for "car"). They
demonstrate the pipeline's decision *logic* is correct — they do not
demonstrate real-world accuracy on actual photos.

### Manual end-to-end check I ran

With the real Week 6 model, over `gunicorn` (matching the Render start
command), I sent real HTTP requests for: a healthy-leaf-colored image
(→ `clear`, 100% confidence, evidence + top-3 shown), a diseased-leaf-colored
image (→ `clear`), car/laptop/person-colored images (→ `non_plant`, disease
list hidden), a dark image (→ `poor_quality`), a truncated JPEG (→ 400
friendly error), a 6 MB upload (→ 413), a request with no file and one with
a `.txt` file (→ 400 friendly errors), and a genuinely ambiguous split-color
image (→ `uncertain`). I also confirmed `/health` never loads the model and
correctly reports `model_loaded: true` only after a real prediction request,
and confirmed the rendered homepage lists all 38 supported classes with no
leftover template syntax.

### Suggested manual test plan for your own real photos

1. A clear, well-lit PlantVillage-style photo of a diseased leaf → expect
   `clear` with a plausible class.
2. A photo of a car, laptop, phone, or person → expect `non_plant`.
3. A deliberately blurry or very dark photo → expect `poor_quality`.
4. A clear photo of a real plant/leaf **not** in the 38 supported classes
   → expect `uncertain` or `out_of_domain`; if you instead see a confident
   `clear` result, that is the documented OOD-layer limitation above, and a
   sign the vegetation gate's threshold may need retuning for your camera.
