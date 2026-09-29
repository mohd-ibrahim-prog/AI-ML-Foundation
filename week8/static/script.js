const form = document.getElementById("upload-form");
const input = document.getElementById("image");
const preview = document.getElementById("preview");
const dropText = document.getElementById("dropzone-text");
const btn = document.getElementById("predict-btn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
  result.hidden = true;
}

input.addEventListener("change", () => {
  errorBox.hidden = true;
  result.hidden = true;
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropText.hidden = true;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  result.hidden = true;

  if (!input.files[0]) {
    showError("Please select an image first.");
    return;
  }

  const data = new FormData();
  data.append("image", input.files[0]);

  btn.disabled = true;
  loading.hidden = false;

  try {
    const response = await fetch("/predict", { method: "POST", body: data });
    let payload = {};
    try { payload = await response.json(); } catch (_) {}

    if (!response.ok) {
      showError(payload.error || "Something went wrong. Please try again.");
      return;
    }

    document.getElementById("result-name").textContent = payload.prediction.label;
    document.getElementById("result-conf").textContent = payload.prediction.confidence.toFixed(2);
    document.getElementById("result-bar").style.width = payload.prediction.confidence + "%";

    const top3 = document.getElementById("top3");
    top3.textContent = "";
    payload.top3.forEach((item) => {
      const row = document.createElement("div");
      const name = document.createElement("span");
      const conf = document.createElement("span");
      name.textContent = item.label;
      conf.textContent = item.confidence.toFixed(2) + "%";
      row.append(name, conf);
      top3.appendChild(row);
    });

    result.hidden = false;
  } catch (err) {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    btn.disabled = false;
    loading.hidden = true;
  }
});
