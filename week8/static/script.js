
// =========================================================
// LEAFGUARD — FRONTEND INTERACTIONS
// Flask backend endpoint: POST /predict
// Form field: image
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    // -----------------------------------------------------
    // ELEMENT REFERENCES
    // -----------------------------------------------------

    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("image");
    const dropzone = document.getElementById("dropzone");

    const emptyUpload = document.getElementById("dropzone-empty");
    const previewContainer = document.getElementById("preview-container");
    const preview = document.getElementById("preview");
    const previewFileName = document.getElementById("preview-file-name");
    const previewFileSize = document.getElementById("preview-file-size");

    const predictButton = document.getElementById("predict-btn");

    const loading = document.getElementById("loading");

    const errorBox = document.getElementById("error");
    const errorText = document.getElementById("error-text");
    const errorRetryButton = document.getElementById("error-retry-btn");

    const clearResult = document.getElementById("result-clear");
    const withheldResult = document.getElementById("result-withheld");

    const resultName = document.getElementById("result-name");
    const resultConfidence = document.getElementById("result-conf");
    const confidenceLabel = document.getElementById("confidence-label");
    const confidenceBar = document.getElementById("result-bar");

    const evidenceList = document.getElementById("evidence-list");
    const top3Container = document.getElementById("top3");

    const withheldReason = document.getElementById("withheld-reason-label");
    const withheldText = document.getElementById("withheld-text");
    const withheldIcon = document.getElementById("withheld-icon");
    const withheldEvidenceWrap = document.getElementById(
        "withheld-evidence-wrap"
    );
    const withheldEvidenceList = document.getElementById(
        "withheld-evidence-list"
    );

    const tryAgainButton = document.getElementById("try-again-btn");

    // -----------------------------------------------------
    // STATE
    // -----------------------------------------------------

    let selectedFile = null;
    let previewURL = null;
    let isSubmitting = false;

    const MAX_FILE_SIZE = 20 * 1024 * 1024;

    const ALLOWED_TYPES = [
        "image/jpeg",
        "image/png"
    ];

    // -----------------------------------------------------
    // GENERAL HELPERS
    // -----------------------------------------------------

    function hide(element) {
        if (element) {
            element.hidden = true;
        }
    }

    function show(element) {
        if (element) {
            element.hidden = false;
        }
    }

    function setText(element, value) {
        if (element) {
            element.textContent =
                value === null || value === undefined
                    ? ""
                    : String(value);
        }
    }

    function safeNumber(value, fallback = 0) {
        const number = Number(value);

        return Number.isFinite(number)
            ? number
            : fallback;
    }

    function formatSize(bytes) {
        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }

        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function normalizeConfidence(value) {
        let confidence = safeNumber(value);

        // Support either 0–1 or 0–100 representation.
        if (confidence > 0 && confidence <= 1) {
            confidence *= 100;
        }

        return Math.max(0, Math.min(100, confidence));
    }

    function resetResults() {
        hide(errorBox);
        hide(clearResult);
        hide(withheldResult);
        hide(loading);

        if (confidenceBar) {
            confidenceBar.style.width = "0%";
        }

        if (evidenceList) {
            evidenceList.replaceChildren();
        }

        if (top3Container) {
            top3Container.replaceChildren();
        }

        if (withheldEvidenceList) {
            withheldEvidenceList.replaceChildren();
        }

        hide(withheldEvidenceWrap);
    }

    function setButtonState(disabled) {
        if (!predictButton) {
            return;
        }

        predictButton.disabled = disabled;

        const label = predictButton.querySelector("span:first-child");

        if (label) {
            label.textContent = disabled
                ? "Analyzing image..."
                : "Analyze image";
        }
    }

    function revokePreviewURL() {
        if (previewURL) {
            URL.revokeObjectURL(previewURL);
            previewURL = null;
        }
    }

    // -----------------------------------------------------
    // FILE VALIDATION
    // -----------------------------------------------------

    function validateFile(file) {
        if (!file) {
            return "Please select an image first.";
        }

        if (!ALLOWED_TYPES.includes(file.type)) {
            return "Please upload a JPG or PNG image.";
        }

        if (file.size === 0) {
            return "The selected file is empty.";
        }

        if (file.size > MAX_FILE_SIZE) {
            return "The image is too large. Please choose a smaller image.";
        }

        return null;
    }

    // -----------------------------------------------------
    // FILE PREVIEW
    // -----------------------------------------------------

    function updatePreview(file) {
        revokePreviewURL();

        previewURL = URL.createObjectURL(file);

        preview.src = previewURL;
        setText(previewFileName, file.name);
        setText(previewFileSize, formatSize(file.size));

        hide(emptyUpload);
        show(previewContainer);

        predictButton.disabled = false;

        resetResults();
    }

    function clearSelectedFile() {
        selectedFile = null;

        if (fileInput) {
            fileInput.value = "";
        }

        revokePreviewURL();

        if (preview) {
            preview.removeAttribute("src");
        }

        hide(previewContainer);
        show(emptyUpload);

        predictButton.disabled = true;

        resetResults();
    }

    function handleFile(file) {
        const validationError = validateFile(file);

        if (validationError) {
            clearSelectedFile();
            showError(validationError);
            return;
        }

        selectedFile = file;

        resetResults();
        updatePreview(file);
    }

    // -----------------------------------------------------
    // FILE INPUT
    // -----------------------------------------------------

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files?.[0];

            if (file) {
                handleFile(file);
            }
        });
    }

    // -----------------------------------------------------
    // DRAG AND DROP
    // -----------------------------------------------------

    if (dropzone) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, event => {
                event.preventDefault();
                event.stopPropagation();

                if (!isSubmitting) {
                    dropzone.classList.add("drag-active");
                }
            });
        });

        ["dragleave", "dragend", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, event => {
                event.preventDefault();
                event.stopPropagation();

                dropzone.classList.remove("drag-active");
            });
        });

        dropzone.addEventListener("drop", event => {
            if (isSubmitting) {
                return;
            }

            const file = event.dataTransfer?.files?.[0];

            if (file) {
                handleFile(file);
            }
        });
    }

    // -----------------------------------------------------
    // ERROR DISPLAY
    // -----------------------------------------------------

    function showError(message) {
        resetResults();

        setText(errorText, message || "An unexpected error occurred.");

        show(errorBox);

        errorBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }

    // -----------------------------------------------------
    // SAFE LIST RENDERING
    // -----------------------------------------------------

    function renderList(container, items) {
        if (!container) {
            return;
        }

        container.replaceChildren();

        if (!Array.isArray(items)) {
            return;
        }

        items.forEach(item => {
            let text;

            if (typeof item === "string") {
                text = item;
            } else if (item && typeof item === "object") {
                text =
                    item.message ??
                    item.reason ??
                    item.description ??
                    item.label ??
                    item.name ??
                    JSON.stringify(item);
            } else {
                text = String(item);
            }

            const li = document.createElement("li");
            li.textContent = text;

            container.appendChild(li);
        });
    }

    // -----------------------------------------------------
    // TOP THREE ALTERNATIVES
    // -----------------------------------------------------

    function renderTopThree(items) {
        if (!top3Container) {
            return;
        }

        top3Container.replaceChildren();

        if (!Array.isArray(items)) {
            return;
        }

        items.slice(0, 3).forEach((item, index) => {
            let name;
            let confidence;

            if (typeof item === "string") {
                name = item;
                confidence = 0;
            } else {
                name =
                    item.label ??
                    item.class ??
                    item.name ??
                    item.prediction ??
                    "Unknown class";

                confidence = normalizeConfidence(
                    item.confidence ??
                    item.probability ??
                    item.score ??
                    0
                );
            }

            const row = document.createElement("div");
            row.className = "top3-item";

            const rank = document.createElement("span");
            rank.className = "top3-rank";
            rank.textContent = String(index + 1);

            const info = document.createElement("div");
            info.className = "top3-info";

            const label = document.createElement("span");
            label.className = "top3-name";
            label.textContent = name;

            const track = document.createElement("div");
            track.className = "top3-track";

            const fill = document.createElement("div");
            fill.className = "top3-fill";
            fill.style.width = `${confidence}%`;

            track.appendChild(fill);
            info.append(label, track);

            const score = document.createElement("span");
            score.className = "top3-score";
            score.textContent = `${confidence.toFixed(1)}%`;

            row.append(rank, info, score);
            top3Container.appendChild(row);
        });
    }

    // -----------------------------------------------------
    // SUCCESS RESULT
    // -----------------------------------------------------

    function showClearResult(data) {
        resetResults();

        const prediction =
            data.prediction ??
            data.class_name ??
            data.label ??
            data.disease ??
            "Unknown";

        const confidence = normalizeConfidence(
            data.confidence ??
            data.prediction_confidence ??
            data.probability ??
            0
        );

        setText(resultName, prediction);
        setText(resultConfidence, confidence.toFixed(1));
        setText(confidenceLabel, confidence.toFixed(1));

        if (confidenceBar) {
            requestAnimationFrame(() => {
                confidenceBar.style.width = `${confidence}%`;
            });
        }

        renderList(
            evidenceList,
            data.evidence ?? data.reasons ?? []
        );

        renderTopThree(
            data.top3 ??
            data.top_3 ??
            data.top_predictions ??
            []
        );

        show(clearResult);

        clearResult.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }

    // -----------------------------------------------------
    // WITHHELD RESULT
    // -----------------------------------------------------

    const WITHHELD_MESSAGES = {
        poor_quality: {
            title: "Image quality is insufficient",
            icon: "!",
            message:
                "The image did not meet the quality requirements. Try a sharper, well-lit image with the leaf clearly visible."
        },

        non_plant: {
            title: "A suitable plant image was not detected",
            icon: "!",
            message:
                "The uploaded image does not appear suitable for this plant-leaf analysis. Please upload a clear plant-leaf image."
        },

        uncertain: {
            title: "The model is not sufficiently confident",
            icon: "?",
            message:
                "The available evidence is not strong enough to provide a reliable result. Try another clear image."
        },

        out_of_domain: {
            title: "Image outside supported classes",
            icon: "!",
            message:
                "The image may not match the categories supported by this model. Please try an image of a supported plant or leaf."
        }
    };

    function showWithheldResult(data) {
        resetResults();

        const reason =
            data.reason ??
            data.status ??
            data.code ??
            "uncertain";

        const details =
            WITHHELD_MESSAGES[reason] ??
            WITHHELD_MESSAGES.uncertain;

        setText(withheldReason, details.title);
        setText(withheldIcon, details.icon);

        const backendMessage =
            data.message ??
            data.explanation ??
            data.detail;

        setText(
            withheldText,
            backendMessage || details.message
        );

        const evidence =
            data.evidence ??
            data.reasons ??
            [];

        if (Array.isArray(evidence) && evidence.length > 0) {
            renderList(withheldEvidenceList, evidence);
            show(withheldEvidenceWrap);
        }

        show(withheldResult);

        withheldResult.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }

    // -----------------------------------------------------
    // BACKEND RESPONSE ROUTING
    // -----------------------------------------------------

    function handleResponse(data) {
        if (!data || typeof data !== "object") {
            throw new Error("The server returned an invalid response.");
        }

        const status = String(
            data.status ??
            data.result ??
            ""
        ).toLowerCase();

        const reason = String(
            data.reason ??
            data.code ??
            ""
        ).toLowerCase();

        const withheldStatuses = [
            "poor_quality",
            "non_plant",
            "uncertain",
            "out_of_domain",
            "withheld"
        ];

        const isWithheld =
            withheldStatuses.includes(status) ||
            withheldStatuses.includes(reason) ||
            data.withheld === true;

        if (isWithheld) {
            showWithheldResult(data);
            return;
        }

        if (
            status === "clear" ||
            status === "success" ||
            status === "ok" ||
            data.prediction !== undefined ||
            data.class_name !== undefined
        ) {
            showClearResult(data);
            return;
        }

        throw new Error(
            data.message ||
            data.error ||
            "The server response could not be understood."
        );
    }

    // -----------------------------------------------------
    // SUBMIT IMAGE TO FLASK
    // -----------------------------------------------------

    async function submitImage(event) {
        event.preventDefault();

        if (isSubmitting) {
            return;
        }

        if (!selectedFile) {
            showError("Please select an image before analyzing.");
            return;
        }

        const validationError = validateFile(selectedFile);

        if (validationError) {
            showError(validationError);
            return;
        }

        isSubmitting = true;

        resetResults();
        show(loading);
        setButtonState(true);

        try {
            const formData = new FormData();

            // IMPORTANT:
            // Keep the original Flask field name.
            formData.append("image", selectedFile);

            // IMPORTANT:
            // Keep the original backend route.
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            let data;

            try {
                data = await response.json();
            } catch {
                throw new Error(
                    "The server returned an invalid response. Please try again."
                );
            }

            if (!response.ok) {
                throw new Error(
                    data.message ||
                    data.error ||
                    `Server error (${response.status}).`
                );
            }

            handleResponse(data);

        } catch (error) {
            console.error("LeafGuard analysis error:", error);

            showError(
                error.message ||
                "Something went wrong while analyzing the image."
            );

        } finally {
            isSubmitting = false;

            hide(loading);
            setButtonState(false);
        }
    }

    if (form) {
        form.addEventListener("submit", submitImage);
    }

    // -----------------------------------------------------
    // TRY AGAIN
    // -----------------------------------------------------

    function resetForAnotherImage() {
        clearSelectedFile();

        if (fileInput) {
            fileInput.click();
        }
    }

    if (tryAgainButton) {
        tryAgainButton.addEventListener(
            "click",
            resetForAnotherImage
        );
    }

    if (errorRetryButton) {
        errorRetryButton.addEventListener(
            "click",
            resetForAnotherImage
        );
    }

    // -----------------------------------------------------
    // CLEAN UP PREVIEW WHEN LEAVING PAGE
    // -----------------------------------------------------

    window.addEventListener("beforeunload", revokePreviewURL);

});
