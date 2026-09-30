/* ================================================================
   LEAFGUARD AI — FRONTEND APPLICATION
   ================================================================ */

const form = document.getElementById("upload-form");

const input = document.getElementById("image");

const dropzone = document.getElementById("dropzone");

const dropzoneEmpty =
    document.getElementById("dropzone-empty");

const previewContainer =
    document.getElementById("preview-container");

const preview =
    document.getElementById("preview");

const previewFileName =
    document.getElementById("preview-file-name");

const previewFileSize =
    document.getElementById("preview-file-size");

const btn =
    document.getElementById("predict-btn");

const loading =
    document.getElementById("loading");


/* ================================================================
   RESULT ELEMENTS
   ================================================================ */

const errorBox =
    document.getElementById("error");

const errorText =
    document.getElementById("error-text");

const errorRetryBtn =
    document.getElementById("error-retry-btn");

const resultClear =
    document.getElementById("result-clear");

const resultWithheld =
    document.getElementById("result-withheld");

const tryAgainBtn =
    document.getElementById("try-again-btn");


/* ================================================================
   WITHHELD RESULT META
   ================================================================ */

const WITHHELD_META = {

    poor_quality: {
        icon: "⌁",
        label: "Image quality is too low"
    },

    non_plant: {
        icon: "×",
        label: "No suitable plant detected"
    },

    uncertain: {
        icon: "?",
        label: "The model is not confident enough"
    },

    out_of_domain: {
        icon: "↗",
        label: "Image appears outside the supported domain"
    }
};


/* ================================================================
   EVIDENCE LABELS
   ================================================================ */

const EVIDENCE_LABELS = {

    "quality.width": (value) =>
        `Image width: ${value}px`,

    "quality.height": (value) =>
        `Image height: ${value}px`,

    "quality.blur_score": (value) =>
        `Sharpness score: ${value}`,

    "quality.brightness": (value) =>
        `Brightness: ${value} / 255`,

    vegetation_ratio: (value) =>
        `Plant/vegetation ratio: ${Math.round(value * 100)}%`,

    top1_confidence: (value) =>
        `Top model confidence: ${value}%`,

    margin: (value) =>
        `Prediction margin: ${value} points`,

    entropy_normalized: (value) =>
        `Prediction spread: ${value}`,

    consistency_ratio: (value) =>
        `Repeated-check agreement: ${Math.round(value * 100)}%`
};


/* ================================================================
   GENERAL UI
   ================================================================ */

function hideAllResults() {

    errorBox.hidden = true;

    resultClear.hidden = true;

    resultWithheld.hidden = true;
}


function resetUpload() {

    input.value = "";

    preview.src = "";

    preview.hidden = true;

    previewContainer.hidden = true;

    dropzoneEmpty.hidden = false;

    btn.disabled = true;

    hideAllResults();

    loading.hidden = true;
}


function showError(message) {

    hideAllResults();

    errorText.textContent =
        message ||
        "Something went wrong. Please try again.";

    errorBox.hidden = false;

    errorBox.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


/* ================================================================
   FILE SIZE
   ================================================================ */

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


/* ================================================================
   FILE VALIDATION
   ================================================================ */

function validateSelectedFile(file) {

    if (!file) {
        return false;
    }

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError(
            "Please upload a JPG or PNG image."
        );

        return false;
    }

    return true;
}


/* ================================================================
   PREVIEW
   ================================================================ */

function showPreview(file) {

    if (!validateSelectedFile(file)) {

        input.value = "";

        btn.disabled = true;

        return;
    }

    const objectUrl =
        URL.createObjectURL(file);

    preview.onload = () => {

        URL.revokeObjectURL(objectUrl);

    };

    preview.src = objectUrl;

    preview.hidden = false;

    previewContainer.hidden = false;

    dropzoneEmpty.hidden = true;

    previewFileName.textContent =
        file.name;

    previewFileSize.textContent =
        `${formatFileSize(file.size)} · Ready for analysis`;

    btn.disabled = false;

    hideAllResults();
}


/* ================================================================
   EVIDENCE RENDERING
   ================================================================ */

function renderEvidenceList(
    listElement,
    evidence
) {

    listElement.textContent = "";

    const rows = [];

    if (!evidence) {
        return false;
    }


    /* Quality */

    if (evidence.quality) {

        if (
            evidence.quality.width !== undefined
        ) {

            rows.push(
                EVIDENCE_LABELS[
                    "quality.width"
                ](
                    evidence.quality.width
                )
            );
        }

        if (
            evidence.quality.height !== undefined
        ) {

            rows.push(
                EVIDENCE_LABELS[
                    "quality.height"
                ](
                    evidence.quality.height
                )
            );
        }

        if (
            evidence.quality.blur_score !== undefined
        ) {

            rows.push(
                EVIDENCE_LABELS[
                    "quality.blur_score"
                ](
                    evidence.quality.blur_score
                )
            );
        }

        if (
            evidence.quality.brightness !== undefined
        ) {

            rows.push(
                EVIDENCE_LABELS[
                    "quality.brightness"
                ](
                    evidence.quality.brightness
                )
            );
        }
    }


    /* Plant detection */

    if (
        evidence.vegetation_ratio !== undefined
    ) {

        rows.push(
            EVIDENCE_LABELS.vegetation_ratio(
                evidence.vegetation_ratio
            )
        );
    }


    /* Model */

    if (
        evidence.top1_confidence !== undefined
    ) {

        rows.push(
            EVIDENCE_LABELS.top1_confidence(
                evidence.top1_confidence
            )
        );
    }

    if (
        evidence.margin !== undefined
    ) {

        rows.push(
            EVIDENCE_LABELS.margin(
                evidence.margin
            )
        );
    }

    if (
        evidence.entropy_normalized !== undefined
    ) {

        rows.push(
            EVIDENCE_LABELS.entropy_normalized(
                evidence.entropy_normalized
            )
        );
    }

    if (
        evidence.consistency_ratio !== undefined
    ) {

        rows.push(
            EVIDENCE_LABELS.consistency_ratio(
                evidence.consistency_ratio
            )
        );
    }


    /* Render */

    rows.forEach((text) => {

        const li =
            document.createElement("li");

        li.textContent = text;

        listElement.appendChild(li);
    });


    return rows.length > 0;
}


/* ================================================================
   CLEAR RESULT
   ================================================================ */

function renderClear(payload) {

    hideAllResults();

    if (
        !payload ||
        !payload.prediction
    ) {

        showError(
            "The server returned an incomplete prediction."
        );

        return;
    }


    const prediction =
        payload.prediction;


    const confidence =
        Number(prediction.confidence) || 0;


    /* Prediction */

    document.getElementById(
        "result-name"
    ).textContent =
        prediction.label ||
        "Unknown";


    document.getElementById(
        "result-conf"
    ).textContent =
        confidence.toFixed(2);


    document.getElementById(
        "confidence-label"
    ).textContent =
        confidence.toFixed(2);


    /* Animate confidence bar */

    const bar =
        document.getElementById(
            "result-bar"
        );

    bar.style.width = "0%";

    requestAnimationFrame(() => {

        bar.style.width =
            `${Math.min(
                Math.max(confidence, 0),
                100
            )}%`;
    });


    /* Evidence */

    const evidenceList =
        document.getElementById(
            "evidence-list"
        );

    renderEvidenceList(
        evidenceList,
        payload.evidence || {}
    );


    /* Top 3 */

    const top3 =
        document.getElementById(
            "top3"
        );

    top3.textContent = "";


    if (
        Array.isArray(payload.top3)
    ) {

        payload.top3.forEach(
            (item) => {

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "top3-row";


                const name =
                    document.createElement(
                        "span"
                    );

                name.className =
                    "top3-name";

                name.textContent =
                    item.label ||
                    "Unknown";


                const confidence =
                    document.createElement(
                        "span"
                    );

                confidence.className =
                    "top3-confidence";

                const value =
                    Number(
                        item.confidence
                    ) || 0;

                confidence.textContent =
                    `${value.toFixed(2)}%`;


                row.append(
                    name,
                    confidence
                );

                top3.appendChild(row);
            }
        );
    }


    resultClear.hidden = false;

    resultClear.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


/* ================================================================
   WITHHELD RESULT
   ================================================================ */

function renderWithheld(payload) {

    hideAllResults();


    const meta =
        WITHHELD_META[
            payload.status
        ] ||
        {
            icon: "?",
            label: "Prediction withheld"
        };


    document.getElementById(
        "withheld-icon"
    ).textContent =
        meta.icon;


    document.getElementById(
        "withheld-reason-label"
    ).textContent =
        meta.label;


    document.getElementById(
        "withheld-text"
    ).textContent =
        payload.message ||
        "The system could not safely determine a result from this image.";


    /* Evidence */

    const evidenceWrap =
        document.getElementById(
            "withheld-evidence-wrap"
        );

    const evidenceList =
        document.getElementById(
            "withheld-evidence-list"
        );


    const hasEvidence =
        renderEvidenceList(
            evidenceList,
            payload.evidence || {}
        );


    evidenceWrap.hidden =
        !hasEvidence;


    resultWithheld.hidden = false;


    resultWithheld.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


/* ================================================================
   INPUT CHANGE
   ================================================================ */

input.addEventListener(
    "change",
    () => {

        const file =
            input.files[0];

        if (!file) {

            resetUpload();

            return;
        }

        showPreview(file);
    }
);


/* ================================================================
   DRAG AND DROP
   ================================================================ */

["dragenter", "dragover"].forEach(
    (eventName) => {

        dropzone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                event.stopPropagation();

                dropzone.classList.add(
                    "drag"
                );
            }
        );
    }
);


["dragleave", "drop"].forEach(
    (eventName) => {

        dropzone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                event.stopPropagation();

                dropzone.classList.remove(
                    "drag"
                );
            }
        );
    }
);


dropzone.addEventListener(
    "drop",
    (event) => {

        const files =
            event.dataTransfer.files;

        if (
            !files ||
            files.length === 0
        ) {
            return;
        }


        const file =
            files[0];


        try {

            const dataTransfer =
                new DataTransfer();

            dataTransfer.items.add(file);

            input.files =
                dataTransfer.files;

        } catch (error) {

            /*
             * Some browsers restrict programmatic
             * file assignment. We can still preview
             * the dropped file directly.
             */

            showPreview(file);

            return;
        }


        showPreview(file);
    }
);


/* ================================================================
   TRY AGAIN
   ================================================================ */

tryAgainBtn.addEventListener(
    "click",
    () => {

        resetUpload();

        dropzone.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        setTimeout(() => {
            input.click();
        }, 250);
    }
);


errorRetryBtn.addEventListener(
    "click",
    () => {

        resetUpload();

        setTimeout(() => {
            input.click();
        }, 250);
    }
);


/* ================================================================
   FORM SUBMISSION
   ================================================================ */

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        hideAllResults();


        const file =
            input.files[0];


        if (!file) {

            showError(
                "Please select an image before starting the analysis."
            );

            return;
        }


        if (
            !validateSelectedFile(file)
        ) {

            return;
        }


        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );


        btn.disabled = true;

        loading.hidden = false;


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            let payload = {};


            try {

                payload =
                    await response.json();

            } catch (jsonError) {

                payload = {};
            }


            if (!response.ok) {

                showError(
                    payload.error ||
                    `Server error (${response.status}). Please try again.`
                );

                return;
            }


            /*
             * CLEAR RESULT
             */

            if (
                payload.status === "clear"
            ) {

                renderClear(payload);

                return;
            }


            /*
             * WITHHELD RESULT
             *
             * This is intentional.
             *
             * The backend can return:
             *
             * poor_quality
             * non_plant
             * uncertain
             * out_of_domain
             */

            renderWithheld(payload);

        } catch (error) {

            console.error(
                "Prediction request failed:",
                error
            );

            showError(
                "The analysis server could not be reached. Make sure Flask is running and try again."
            );

        } finally {

            loading.hidden = true;

            btn.disabled = false;
        }
    }
);


/* ================================================================
   INITIAL STATE
   ================================================================ */

hideAllResults();

loading.hidden = true;

btn.disabled =
    !input.files.length;
