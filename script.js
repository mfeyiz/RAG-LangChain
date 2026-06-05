const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendButton = document.getElementById("sendButton");
const chatForm = document.getElementById("chatForm");
const statusBadge = document.getElementById("statusBadge");
const progressBar = document.getElementById("progressBar");
const progressPercent = document.getElementById("progressPercent");
const progressLabel = document.getElementById("progressLabel");
const flowStateText = document.getElementById("flowStateText");
const currentAgentLabel = document.getElementById("currentAgentLabel");
const eventLog = document.getElementById("eventLog");
const retrievalList = document.getElementById("retrievalList");
const retrievalLabel = document.getElementById("retrievalLabel");
const activeAgentMetric = document.getElementById("activeAgentMetric");
const documentMetric = document.getElementById("documentMetric");
const elapsedMetric = document.getElementById("elapsedMetric");
const executeButton = document.getElementById("executeButton");
const newSessionButton = document.getElementById("newSessionButton");
const heroMeter = document.querySelector(".hero-meter");
const agentCards = Array.from(document.querySelectorAll(".agent-card"));
const connectors = Array.from(document.querySelectorAll(".flow-connector"));

const API_URL = "/ask";
const AGENT_ORDER = ["supervisor", "researcher", "writer", "reviewer"];
const AGENT_PROGRESS = {
    supervisor: 18,
    researcher: 45,
    writer: 74,
    reviewer: 92,
};
const AGENT_LABELS = {
    supervisor: "Supervisor",
    researcher: "Researcher",
    writer: "Writer",
    reviewer: "Reviewer",
};
const AGENT_MESSAGES = {
    supervisor: "Yonlendirme karari veriliyor",
    researcher: "Vektor veritabani taraniyor",
    writer: "Kanitlara dayali yanit yaziliyor",
    reviewer: "Cevap kalite kontrolden geciyor",
};

let elapsedTimer = null;
let startedAt = 0;
let visitedAgents = new Set();

async function sendMessage() {
    const message = userInput.value.trim();

    if (!message || sendButton.disabled) return;

    resetRunState();
    addMessage(message, "user");
    userInput.value = "";
    autoResize();
    setControlsDisabled(true);

    setStatus("Running", "running");
    updateProgress(8, "Istek alindi", "Input staging");
    logEvent("request", "Kullanici istegi alindi. Graph calistiriliyor.");
    startElapsedTimer();

    const typingIndicator = addTypingIndicator();
    const botMessageDiv = createBotMessage();
    const contentDiv = botMessageDiv.querySelector(".message-content");
    let fullText = "";
    let completed = false;

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
            },
            body: JSON.stringify({ query: message }),
        });

        if (!response.ok || !response.body) {
            let detail = "";
            try {
                const payload = await response.json();
                detail = payload?.error ? ` ${payload.error}` : "";
            } catch {
                detail = "";
            }
            throw new Error(`API istegi basarisiz oldu.${detail}`);
        }

        typingIndicator.remove();

        for await (const event of readSseEvents(response.body)) {
            if (event.event === "agent_update") {
                const payload = safeJson(event.data);
                if (payload && payload.agent) {
                    markAgentActive(payload.agent);
                }
            }

            if (event.event === "search_results") {
                const results = safeJson(event.data) || [];
                showSearchResults(results);
                logEvent("retrieval", `${results.length} dokuman skoru alindi.`);
            }

            if (event.event === "message") {
                fullText += event.data;
                contentDiv.textContent = fullText;
                scrollToBottom();
            }

            if (event.event === "done") {
                markComplete();
                completed = true;
            }

            if (event.event === "error") {
                const payload = safeJson(event.data);
                throw new Error(payload?.error || "Bilinmeyen stream hatasi.");
            }
        }

        if (!fullText.trim()) {
            contentDiv.textContent = "Akis tamamlandi fakat yanit metni uretilmedi.";
        }

        if (!completed) {
            markComplete();
        }
    } catch (error) {
        typingIndicator.remove();
        if (!fullText.trim()) {
            contentDiv.textContent = `Uzgunum, bir hata olustu. ${error.message || "Lutfen tekrar deneyin."}`;
        }
        markError(error.message);
        console.error(error);
    } finally {
        stopElapsedTimer();
        setControlsDisabled(false);
        userInput.focus();
    }
}

async function* readSseEvents(stream) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || "";

        for (const frame of frames) {
            const parsed = parseSseFrame(frame);
            if (parsed) yield parsed;
        }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
        const parsed = parseSseFrame(buffer);
        if (parsed) yield parsed;
    }
}

function parseSseFrame(frame) {
    let event = "message";
    const dataLines = [];

    frame.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) {
            event = line.slice(6).trim();
        }

        if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    });

    if (!dataLines.length) return null;

    return {
        event,
        data: dataLines.join("\n"),
    };
}

function markAgentActive(agent) {
    const normalizedAgent = AGENT_ORDER.includes(agent) ? agent : "supervisor";
    const activeIndex = AGENT_ORDER.indexOf(normalizedAgent);
    visitedAgents.add(normalizedAgent);

    agentCards.forEach((card) => {
        const cardAgent = card.dataset.agent;
        const cardIndex = AGENT_ORDER.indexOf(cardAgent);
        card.classList.remove("is-active", "has-error");
        card.classList.toggle("is-complete", cardIndex < activeIndex || visitedAgents.has(cardAgent) && cardAgent !== normalizedAgent);
        card.classList.toggle("is-active", cardAgent === normalizedAgent);
    });

    connectors.forEach((connector, index) => {
        connector.classList.toggle("is-active", index < activeIndex);
    });

    const statusNode = document.getElementById(`${normalizedAgent}Status`);
    if (statusNode) {
        statusNode.textContent = AGENT_MESSAGES[normalizedAgent];
    }

    activeAgentMetric.textContent = String(visitedAgents.size);
    currentAgentLabel.textContent = AGENT_LABELS[normalizedAgent];
    setStatus(`${AGENT_LABELS[normalizedAgent]} working`, "running");
    updateProgress(AGENT_PROGRESS[normalizedAgent], AGENT_MESSAGES[normalizedAgent], normalizedAgent);
    logEvent(normalizedAgent, AGENT_MESSAGES[normalizedAgent]);
}

function markComplete() {
    agentCards.forEach((card) => {
        card.classList.remove("is-active", "has-error");
        card.classList.add("is-complete");
    });
    connectors.forEach((connector) => connector.classList.remove("is-active"));
    updateProgress(100, "Cevap hazir", "Complete");
    currentAgentLabel.textContent = "Tamamlandi";
    setStatus("Complete", "complete");
    logEvent("done", "Akis tamamlandi, yanit teslim edildi.");
}

function markError(message) {
    agentCards.forEach((card) => card.classList.remove("is-active"));
    const lastAgent = Array.from(visitedAgents).pop();
    const errorCard = lastAgent ? document.querySelector(`[data-agent="${lastAgent}"]`) : null;
    if (errorCard) errorCard.classList.add("has-error");
    updateProgress(100, "Hata olustu", "Error");
    currentAgentLabel.textContent = "Hata";
    setStatus("Error", "error");
    logEvent("error", message || "Akis hata ile sonlandi.");
}

function resetRunState() {
    visitedAgents = new Set();
    agentCards.forEach((card) => card.classList.remove("is-active", "is-complete", "has-error"));
    connectors.forEach((connector) => connector.classList.remove("is-active"));
    AGENT_ORDER.forEach((agent) => {
        const statusNode = document.getElementById(`${agent}Status`);
        if (statusNode) statusNode.textContent = AGENT_MESSAGES[agent];
    });
    retrievalList.innerHTML = `
        <div class="empty-state">
            <span class="material-symbols-outlined">database</span>
            <p>Yeni bir sorgu calistiginda kanitlar burada listelenir.</p>
        </div>
    `;
    retrievalLabel.textContent = "Bos";
    activeAgentMetric.textContent = "0";
    documentMetric.textContent = "0";
    elapsedMetric.textContent = "0.0s";
}

function updateProgress(percent, label, stateText) {
    const safePercent = Math.max(0, Math.min(100, percent));
    progressBar.style.width = `${safePercent}%`;
    progressPercent.textContent = `${safePercent}%`;
    progressLabel.textContent = label;
    flowStateText.textContent = stateText;
    if (heroMeter) {
        heroMeter.style.setProperty("--meter-progress", `${safePercent}%`);
    }
}

function setStatus(label, mode) {
    statusBadge.className = `status-pill ${mode || ""}`.trim();
    statusBadge.innerHTML = `<span class="status-dot"></span>${label}`;
}

function showSearchResults(results) {
    retrievalList.innerHTML = "";
    documentMetric.textContent = String(results.length);
    retrievalLabel.textContent = `${results.length} dokuman`;

    if (!results.length) {
        retrievalList.innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">travel_explore</span>
                <p>Bu sorgu icin dokuman bulunamadi.</p>
            </div>
        `;
        return;
    }

    results.forEach((result, index) => {
        const card = document.createElement("article");
        card.className = `retrieval-card ${result.relevant ? "relevant" : "not-relevant"}`;

        const content = result.content || "";
        const score = typeof result.score === "number" ? result.score.toFixed(4) : "n/a";
        const denseScore = typeof result.dense_score === "number" ? result.dense_score.toFixed(3) : "0.000";
        const bm25Score = typeof result.bm25_score === "number" ? result.bm25_score.toFixed(3) : "0.000";
        const title = result.title || `Doc ${index + 1}`;
        const source = result.source || "unknown";

        card.innerHTML = `
            <div class="retrieval-head">
                <strong>${escapeHtml(title)}</strong>
                <span class="score-pill">score ${score}</span>
                <span class="relevance-pill">dense ${denseScore}</span>
                <span class="relevance-pill">bm25 ${bm25Score}</span>
            </div>
            <small>${escapeHtml(source)}</small>
            <p>${escapeHtml(content)}</p>
        `;
        retrievalList.appendChild(card);
    });
}

function addMessage(text, type) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = type === "user" ? "YOU" : "AI";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function createBotMessage() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";

    const content = document.createElement("div");
    content.className = "message-content";

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function addTypingIndicator() {
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";
    messageDiv.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";

    const content = document.createElement("div");
    content.className = "message-content typing-indicator";
    content.innerHTML = "<span></span><span></span><span></span>";

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function logEvent(type, message) {
    if (!eventLog) return;

    const item = document.createElement("div");
    item.className = `event-item ${type === "error" ? "error" : type === "done" ? "done" : ""}`;
    const time = new Date().toLocaleTimeString("tr-TR", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
    item.innerHTML = `<span>${time} | ${escapeHtml(type)}</span><p>${escapeHtml(message)}</p>`;
    eventLog.appendChild(item);
    eventLog.scrollTop = eventLog.scrollHeight;
}

function startElapsedTimer() {
    stopElapsedTimer();
    startedAt = performance.now();
    elapsedTimer = window.setInterval(() => {
        const seconds = (performance.now() - startedAt) / 1000;
        elapsedMetric.textContent = `${seconds.toFixed(1)}s`;
    }, 100);
}

function stopElapsedTimer() {
    if (elapsedTimer) {
        window.clearInterval(elapsedTimer);
        elapsedTimer = null;
    }
}

function setControlsDisabled(disabled) {
    sendButton.disabled = disabled;
    executeButton.disabled = disabled;
    userInput.disabled = disabled;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = `${userInput.scrollHeight}px`;
}

function safeJson(value) {
    try {
        return JSON.parse(value);
    } catch {
        return null;
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
});

executeButton.addEventListener("click", () => sendMessage());

newSessionButton.addEventListener("click", () => {
    chatMessages.innerHTML = `
        <div class="message bot-message">
            <div class="message-avatar">AI</div>
            <div class="message-content">
                Yeni oturum hazir. Sorunuzu yazin; ajan akisindaki her adimi canli olarak gosterecegim.
            </div>
        </div>
    `;
    eventLog.innerHTML = "";
    resetRunState();
    updateProgress(0, "Hazir", "Idle");
    setStatus("Idle", "");
    logEvent("idle", "Yeni oturum baslatildi.");
    userInput.focus();
});

userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener("input", autoResize);

window.addEventListener("load", () => {
    scrollToBottom();
    updateProgress(0, "Hazir", "Idle");
    logEvent("idle", "Sistem hazir. Yeni bir istek bekleniyor.");
});
