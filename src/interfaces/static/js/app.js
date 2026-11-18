const questionnaire = document.getElementById("questionnaire");
const questionFields = Array.from(
    questionnaire.querySelectorAll("select[required]")
);
const answeredCount = document.getElementById("answeredCount");
const totalCount = document.getElementById("totalCount");
const progressTrack = document.getElementById("progressTrack");
const progressBar = document.getElementById("progressBar");
const progressMessage = document.getElementById("progressMessage");
const submitButton = document.getElementById("submitButton");
const submitHint = document.getElementById("submitHint");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const loadingState = document.getElementById("loadingState");
const resultsPanel = document.getElementById("resultsPanel");
const resultsGrid = document.getElementById("resultsGrid");
const navigationTabs = Array.from(
    document.querySelectorAll(".sidebar-nav [role='tab']")
);
const viewPanels = Array.from(document.querySelectorAll(".view-panel[role='tabpanel']"));
const activeSectionName = document.getElementById("activeSectionName");
const historyCount = document.getElementById("historyCount");
const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");
const historyClearAll = document.getElementById("historyClearAll");
const recordDialog = document.getElementById("recordDialog");
const recordDialogTitle = document.getElementById("recordDialogTitle");
const recordDialogDate = document.getElementById("recordDialogDate");
const recordDialogResults = document.getElementById("recordDialogResults");
const recordDialogAnswers = document.getElementById("recordDialogAnswers");
const recordDialogClose = document.getElementById("recordDialogClose");
const serverWarning = document.getElementById("serverWarning");
const HISTORY_STORAGE_KEY = "equilibre-bilans-v1";
const sectionNames = {
    questionnaire: "Questionnaire",
    properties: "Propriétés",
    history: "Historique",
    models: "Modèles",
    charts: "Graphiques",
};
let sessionHistory = [];
let historyStorageReady = true;

if (serverWarning) {
    serverWarning.hidden = window.location.protocol !== "file:";
}
totalCount.textContent = questionFields.length;
progressTrack.setAttribute("aria-valuemax", questionFields.length);

function updateProgress() {
    const completed = questionFields.filter((field) => field.value !== "").length;
    const percentage = (completed / questionFields.length) * 100;
    const isComplete = completed === questionFields.length;

    answeredCount.textContent = completed;
    progressBar.style.width = `${percentage}%`;
    progressTrack.setAttribute("aria-valuenow", completed);
    submitButton.disabled = !isComplete;

    progressMessage.textContent = isComplete
        ? "C'est complet. Tu peux découvrir ton bilan."
        : "Chaque réponse te rapproche du bilan.";
    submitHint.textContent = isComplete
        ? "Tout est prêt — tes réponses seront analysées par les modèles."
        : `Réponds aux ${questionFields.length} questions pour continuer.`;
}

function createProbabilityRow(label, probability) {
    const percentage = Number(probability);
    if (!Number.isFinite(percentage)) {
        throw new Error("Les probabilités reçues ne sont pas valides.");
    }

    const row = document.createElement("div");
    row.className = "probability-row";

    const meta = document.createElement("div");
    meta.className = "probability-meta";

    const name = document.createElement("span");
    name.textContent = label;

    const score = document.createElement("strong");
    score.textContent = `${percentage.toFixed(2)} %`;
    meta.append(name, score);

    const track = document.createElement("div");
    track.className = "probability-track";
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${label} : ${percentage.toFixed(2)} %`);

    const fill = document.createElement("div");
    const isRisk = label === "At Risk / Depressed";
    fill.className = `probability-fill${isRisk ? " probability-fill--risk" : ""}`;
    fill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
    track.appendChild(fill);

    row.append(meta, track);
    return row;
}

function createModelCard(modelName, modelData) {
    if (
        !modelData
        || typeof modelData.prediction !== "string"
        || !modelData.probabilities
        || typeof modelData.probabilities !== "object"
        || Array.isArray(modelData.probabilities)
        || !Object.prototype.hasOwnProperty.call(
            modelData.probabilities,
            modelData.prediction
        )
    ) {
        throw new Error("Le format des résultats reçus est incomplet.");
    }

    const probabilityValues = Object.values(modelData.probabilities).map(Number);
    if (
        probabilityValues.length === 0
        || probabilityValues.some(
            (value) => !Number.isFinite(value) || value < 0 || value > 100
        )
    ) {
        throw new Error("Les probabilités reçues ne sont pas valides.");
    }

    const card = document.createElement("article");
    card.className = "result-card";

    const header = document.createElement("div");
    header.className = "result-card-head";

    const icon = document.createElement("span");
    icon.className = `model-icon${modelName === "Logistic Regression" ? " model-icon--logistic" : ""}`;
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = modelName === "SVM" ? "S" : "LR";

    const modelTitle = document.createElement("p");
    modelTitle.textContent = modelName === "SVM" ? "Support Vector Machine" : modelName;

    const subtitle = document.createElement("small");
    subtitle.textContent = "Modèle de classification";
    modelTitle.appendChild(subtitle);
    header.append(icon, modelTitle);

    const prediction = document.createElement("div");
    const predictionLabel = modelData.prediction;
    const stable = predictionLabel === "Stable / Not Depressed";
    prediction.className = `result-label ${stable ? "result-label--stable" : "result-label--risk"}`;
    prediction.textContent = predictionLabel;

    const probabilityTitle = document.createElement("p");
    probabilityTitle.className = "probability-title";
    probabilityTitle.textContent = "Probabilités estimées";

    const probabilityRows = Object.entries(modelData.probabilities).map(
        ([label, value]) => createProbabilityRow(label, value)
    );
    if (probabilityRows.length === 0) {
        throw new Error("Aucune probabilité n'a été fournie par le modèle.");
    }

    card.append(header, prediction, probabilityTitle, ...probabilityRows);
    return card;
}

function displayResults(data) {
    if (!data || !data.SVM || !data["Logistic Regression"]) {
        throw new Error("La réponse du serveur ne contient pas les deux résultats.");
    }

    resultsGrid.replaceChildren(
        createModelCard("SVM", data.SVM),
        createModelCard("Logistic Regression", data["Logistic Regression"])
    );
    resultsPanel.hidden = false;
}

function changeView(viewName) {
    const activeTab = navigationTabs.find((tab) => tab.dataset.view === viewName);
    if (!activeTab) {
        return;
    }

    navigationTabs.forEach((tab) => {
        const isActive = tab === activeTab;
        tab.classList.toggle("is-active", isActive);
        tab.setAttribute("aria-selected", String(isActive));
        tab.tabIndex = isActive ? 0 : -1;
    });

    viewPanels.forEach((panel) => {
        const isActive = panel.id === `view-${viewName}`;
        panel.hidden = !isActive;
        panel.tabIndex = isActive ? 0 : -1;
    });

    activeSectionName.textContent = sectionNames[viewName];
}

function createHistoryEntry(record, index) {
    const entry = document.createElement("article");
    entry.className = "history-entry";

    const openButton = document.createElement("button");
    openButton.className = "history-entry-open";
    openButton.type = "button";
    openButton.setAttribute("aria-haspopup", "dialog");
    openButton.setAttribute("aria-controls", "recordDialog");
    openButton.setAttribute(
        "aria-label",
        `Voir le détail du bilan n° ${sessionHistory.length - index}`
    );

    const deleteButton = document.createElement("button");
    deleteButton.className = "history-delete-button";
    deleteButton.type = "button";
    deleteButton.textContent = "Supprimer";
    deleteButton.setAttribute(
        "aria-label",
        `Supprimer le bilan n° ${sessionHistory.length - index}`
    );

    const content = document.createElement("div");
    content.className = "history-entry-content";

    const details = document.createElement("div");
    details.className = "history-entry-main";

    const title = document.createElement("strong");
    title.textContent = `Bilan n° ${sessionHistory.length - index}`;

    const date = document.createElement("small");
    date.textContent = new Intl.DateTimeFormat("fr-FR", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date(record.date));
    details.append(title, date);

    const predictions = document.createElement("div");
    predictions.className = "history-predictions";

    for (const [modelName, modelData] of Object.entries(record.results)) {
        const result = document.createElement("span");
        const prediction = modelData.prediction;
        const isRisk = prediction === "At Risk / Depressed";
        result.className = `history-pill${isRisk ? " history-pill--risk" : ""}`;
        result.textContent = `${modelName === "SVM" ? "SVM" : "Régression logistique"} : ${prediction}`;
        predictions.appendChild(result);
    }

    content.append(details, predictions);
    openButton.appendChild(content);
    entry.append(openButton, deleteButton);
    openButton.addEventListener("click", () => openRecord(record, index));
    deleteButton.addEventListener("click", () => deleteHistoryRecord(index));
    return entry;
}

function renderHistory() {
    historyCount.textContent = sessionHistory.length;
    historyClearAll.hidden = sessionHistory.length === 0;
    historyList.replaceChildren();

    if (sessionHistory.length === 0) {
        historyList.appendChild(historyEmpty);
        return;
    }

    sessionHistory.forEach((record, index) => {
        historyList.appendChild(createHistoryEntry(record, index));
    });
}

function saveSessionHistory() {
    try {
        window.localStorage.setItem(
            HISTORY_STORAGE_KEY,
            JSON.stringify(sessionHistory)
        );
        return true;
    } catch {
        historyStorageReady = false;
        errorText.textContent =
            "La modification de l'historique n'a pas pu être enregistrée dans ce navigateur.";
        errorMessage.hidden = false;
        return false;
    }
}

function deleteHistoryRecord(index) {
    if (!window.confirm("Supprimer ce bilan de ton historique ?")) {
        return;
    }

    sessionHistory.splice(index, 1);
    renderHistory();
    saveSessionHistory();
}

historyClearAll.addEventListener("click", () => {
    if (!window.confirm("Supprimer tous les bilans enregistrés ?")) {
        return;
    }

    sessionHistory = [];
    renderHistory();
    saveSessionHistory();
});

function openRecord(record, index) {
    recordDialogTitle.textContent = `Bilan n° ${sessionHistory.length - index}`;
    recordDialogDate.textContent = new Intl.DateTimeFormat("fr-FR", {
        dateStyle: "full",
        timeStyle: "short",
    }).format(new Date(record.date));

    recordDialogResults.replaceChildren(
        createModelCard("SVM", record.results.SVM),
        createModelCard("Logistic Regression", record.results["Logistic Regression"])
    );

    const answers = questionFields.map((field) => {
        const answer = record.answers[field.name];
        const label = field.labels[0].cloneNode(true);
        const number = label.querySelector("span");
        if (number) {
            number.remove();
        }

        const answerLabel = document.createElement("dt");
        answerLabel.textContent = label.textContent.trim();

        const answerValue = document.createElement("dd");
        const option = Array.from(field.options).find(
            (item) => item.value === answer
        );
        answerValue.textContent = option ? option.textContent.trim() : answer;

        const item = document.createElement("div");
        item.className = "record-answer";
        item.append(answerLabel, answerValue);
        return item;
    });
    recordDialogAnswers.replaceChildren(...answers);
    recordDialog.showModal();
}

function isValidStoredRecord(record) {
    return Boolean(
        record
        && typeof record.date === "string"
        && Number.isFinite(Date.parse(record.date))
        && record.answers
        && typeof record.answers === "object"
        && !Array.isArray(record.answers)
        && questionFields.every(
            (field) => typeof record.answers[field.name] === "string"
        )
        && record.results
        && ["SVM", "Logistic Regression"].every((modelName) => {
            const model = record.results[modelName];
            return Boolean(
                model
                && typeof model.prediction === "string"
                && model.probabilities
                && typeof model.probabilities === "object"
                && !Array.isArray(model.probabilities)
                && Object.keys(model.probabilities).length > 0
                && Object.prototype.hasOwnProperty.call(
                    model.probabilities,
                    model.prediction
                )
                && Object.values(model.probabilities).every((value) =>
                    Number.isFinite(Number(value))
                    && Number(value) >= 0
                    && Number(value) <= 100
                )
            );
        })
    );
}

function loadSessionHistory() {
    const savedHistory = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (savedHistory === null) {
        return [];
    }

    const records = JSON.parse(savedHistory);
    if (!Array.isArray(records) || !records.every(isValidStoredRecord)) {
        throw new Error("Le format de l'historique enregistré est invalide.");
    }
    return records;
}

function addSessionRecord(data) {
    const record = {
        date: new Date().toISOString(),
        answers: Object.fromEntries(new FormData(questionnaire).entries()),
        results: data,
    };
    sessionHistory.unshift(record);
    renderHistory();
    saveSessionHistory();
}

navigationTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => changeView(tab.dataset.view));
    tab.addEventListener("keydown", (event) => {
        let nextIndex = index;
        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
            nextIndex = (index + 1) % navigationTabs.length;
        } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
            nextIndex = (index - 1 + navigationTabs.length) % navigationTabs.length;
        } else if (event.key === "Home") {
            nextIndex = 0;
        } else if (event.key === "End") {
            nextIndex = navigationTabs.length - 1;
        } else {
            return;
        }

        event.preventDefault();
        navigationTabs[nextIndex].focus();
        changeView(navigationTabs[nextIndex].dataset.view);
    });
});

document.querySelectorAll("[data-open-view]").forEach((button) => {
    button.addEventListener("click", () => changeView(button.dataset.openView));
});

recordDialogClose.addEventListener("click", () => recordDialog.close());
recordDialog.addEventListener("click", (event) => {
    if (event.target === recordDialog) {
        recordDialog.close();
    }
});

questionnaire.addEventListener("change", () => {
    updateProgress();
    resultsPanel.hidden = true;
    errorMessage.hidden = true;
});

questionnaire.addEventListener("reset", () => {
    window.setTimeout(() => {
        updateProgress();
        errorMessage.hidden = true;
        loadingState.hidden = true;
        resultsPanel.hidden = true;
        resultsGrid.replaceChildren();
    }, 0);
});

questionnaire.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorMessage.hidden = true;
    resultsPanel.hidden = true;

    if (window.location.protocol === "file:") {
        errorText.textContent =
            "La page est ouverte directement comme fichier. Lance `python app.py` à la racine du projet, puis ouvre http://127.0.0.1:5000/.";
        errorMessage.hidden = false;
        errorMessage.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
    }

    if (!questionnaire.reportValidity()) {
        return;
    }

    submitButton.disabled = true;
    loadingState.hidden = false;
    questionnaire.setAttribute("aria-busy", "true");
    loadingState.scrollIntoView({ behavior: "smooth", block: "center" });
    const minimumLoadingTime = new Promise((resolve) => {
        window.setTimeout(resolve, 3000);
    });

    try {
        const answers = Object.fromEntries(new FormData(questionnaire).entries());
        let response;
        try {
            response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(answers),
            });
        } catch {
            throw new Error(
                "Impossible de joindre Flask. Lance `python app.py`, puis utilise http://127.0.0.1:5000/."
            );
        }

        let responseData;
        try {
            responseData = await response.json();
        } catch {
            throw new Error("Le serveur a renvoyé une réponse illisible.");
        }

        if (!response.ok) {
            throw new Error(responseData.error || "L'analyse n'a pas abouti.");
        }

        await minimumLoadingTime;
        displayResults(responseData);
        addSessionRecord(responseData);
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        errorText.textContent = error.message || "Une erreur inattendue est survenue.";
        errorMessage.hidden = false;
        errorMessage.scrollIntoView({ behavior: "smooth", block: "center" });
    } finally {
        loadingState.hidden = true;
        questionnaire.removeAttribute("aria-busy");
        updateProgress();
    }
});

try {
    sessionHistory = loadSessionHistory();
} catch (error) {
    historyStorageReady = false;
    errorText.textContent =
        `L'historique enregistré n'a pas pu être chargé : ${error.message}`;
    errorMessage.hidden = false;
}
renderHistory();
updateProgress();
