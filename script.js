/* ── DOM refs ────────────────────────────────────────────────── */
const chatMessages      = document.getElementById("chatMessages");
const userInput         = document.getElementById("userInput");
const sendButton        = document.getElementById("sendButton");
const chatForm          = document.getElementById("chatForm");
const statusBadge       = document.getElementById("statusBadge");
const progressBar       = document.getElementById("progressBar");
const progressPercent   = document.getElementById("progressPercent");
const progressLabel     = document.getElementById("progressLabel");
const flowStateText     = document.getElementById("flowStateText");
const currentAgentLabel = document.getElementById("currentAgentLabel");
const eventLog          = document.getElementById("eventLog");
const retrievalList     = document.getElementById("retrievalList");
const retrievalLabel    = document.getElementById("retrievalLabel");
const activeAgentMetric = document.getElementById("activeAgentMetric");
const documentMetric    = document.getElementById("documentMetric");
const elapsedMetric     = document.getElementById("elapsedMetric");
const newSessionButton  = document.getElementById("newSessionButton");
const uploadButton      = document.getElementById("uploadButton");
const uploadModal       = document.getElementById("uploadModal");
const uploadModalClose  = document.getElementById("uploadModalClose");
const uploadArea        = document.getElementById("uploadArea");
const fileInput         = document.getElementById("fileInput");
const uploadStatus      = document.getElementById("uploadStatus");
const suggestionsDropdown = document.getElementById("suggestionsDropdown");
const heroMeter         = document.querySelector(".hero-meter");
const agentCards        = Array.from(document.querySelectorAll(".agent-card"));
const connectors        = Array.from(document.querySelectorAll(".flow-connector"));

/* ── Constants ───────────────────────────────────────────────── */
const API_URL          = "/ask";
const SUGGESTIONS_URL  = "/suggestions";
const FEEDBACK_URL     = "/feedback";
const UPLOAD_URL       = "/upload";

const AGENT_ORDER    = ["supervisor", "researcher", "writer", "reviewer"];
const AGENT_PROGRESS = { supervisor: 18, researcher: 45, writer: 74, reviewer: 92 };
const AGENT_LABELS   = { supervisor: "Supervisor", researcher: "Researcher", writer: "Writer", reviewer: "Reviewer" };
const AGENT_MESSAGES = {
    supervisor: "Yönlendirme kararı veriliyor",
    researcher: "Vektör veritabanı taranıyor",
    writer:     "Kanıtlara dayalı yanıt yazılıyor",
    reviewer:   "Cevap kalite kontrolden geçiyor",
};

/* ── Session state ───────────────────────────────────────────── */
let currentSessionId = localStorage.getItem("rag_session_id") || generateUUID();
localStorage.setItem("rag_session_id", currentSessionId);
let currentTraceId = "";

let elapsedTimer  = null;
let startedAt     = 0;
let visitedAgents = new Set();

/* ── Suggestions state ───────────────────────────────────────── */
let suggestDebounce = null;
let activeSuggestionIndex = -1;
let currentSuggestions = [];

/* ══════════════════════════════════════════════════════════════
   SEND MESSAGE
═══════════════════════════════════════════════════════════════ */
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || sendButton.disabled) return;

    hideSuggestions();
    resetRunState();
    const welcome = chatMessages.querySelector(".welcome");
    if (welcome) welcome.remove();
    addMessage(message, "user");
    userInput.value = "";
    autoResize();
    setControlsDisabled(true);

    setStatus("Running", "running");
    updateProgress(8, "İstek alındı", "Input staging");
    logEvent("request", "Kullanıcı isteği alındı. Graph çalıştırılıyor.");
    startElapsedTimer();

    const typingIndicator = addTypingIndicator();
    const botMessageDiv   = createBotMessage();
    const contentDiv      = botMessageDiv.querySelector(".message-content");
    let fullText          = "";
    let streamingStarted  = false;
    let feedbackAdded     = false;
    let completed         = false;

    try {
        const response = await fetch(API_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
            body:    JSON.stringify({ query: message, session_id: currentSessionId }),
        });

        if (!response.ok || !response.body) {
            let detail = "";
            try { const p = await response.json(); detail = p?.error ? ` ${p.error}` : ""; } catch {}
            throw new Error(`API isteği başarısız oldu.${detail}`);
        }

        typingIndicator.remove();

        for await (const event of readSseEvents(response.body)) {
            /* session / trace ids */
            if (event.event === "session_info") {
                const info = safeJson(event.data);
                if (info?.session_id) { currentSessionId = info.session_id; localStorage.setItem("rag_session_id", currentSessionId); }
                if (info?.trace_id)   currentTraceId = info.trace_id;
            }

            /* agent progress */
            if (event.event === "agent_update") {
                const payload = safeJson(event.data);
                if (payload?.agent) markAgentActive(payload.agent);
            }

            /* search results */
            if (event.event === "search_results") {
                const results = safeJson(event.data) || [];
                showSearchResults(results);
                logEvent("retrieval", `${results.length} doküman skoru alındı.`);
            }

            /* streaming tokens from writer */
            if (event.event === "token") {
                streamingStarted = true;
                fullText += event.data;
                contentDiv.textContent = fullText;
                contentDiv.classList.add("is-streaming");
                scrollToBottom();
            }

            /* final complete message */
            if (event.event === "message") {
                streamingStarted = true;
                contentDiv.classList.remove("is-streaming");
                fullText = event.data;
                contentDiv.textContent = fullText;
                scrollToBottom();

                if (!feedbackAdded) {
                    addFeedbackRow(botMessageDiv, message);
                    feedbackAdded = true;
                }
            }

            if (event.event === "done") {
                contentDiv.classList.remove("is-streaming");
                if (!feedbackAdded && fullText) {
                    addFeedbackRow(botMessageDiv, message);
                    feedbackAdded = true;
                }
                markComplete();
                completed = true;
            }

            if (event.event === "error") {
                const payload = safeJson(event.data);
                throw new Error(payload?.error || "Bilinmeyen stream hatası.");
            }
        }

        if (!fullText.trim()) {
            contentDiv.textContent = "Akış tamamlandı fakat yanıt metni üretilmedi.";
        }
        if (!completed) markComplete();

    } catch (error) {
        typingIndicator.remove();
        contentDiv.classList.remove("is-streaming");
        if (!fullText.trim()) {
            contentDiv.textContent = `Üzgünüm, bir hata oluştu. ${error.message || "Lütfen tekrar deneyin."}`;
        }
        markError(error.message);
        console.error(error);
    } finally {
        stopElapsedTimer();
        setControlsDisabled(false);
        userInput.focus();
    }
}

/* ══════════════════════════════════════════════════════════════
   FEEDBACK
═══════════════════════════════════════════════════════════════ */
function addFeedbackRow(botMessageDiv, query) {
    const row = document.createElement("div");
    row.className = "feedback-row";

    const upBtn   = makeFeedbackBtn("thumb_up",   "İyi yanıt",  1);
    const downBtn = makeFeedbackBtn("thumb_down", "Kötü yanıt", -1);

    async function vote(btn, other, rating) {
        if (btn.classList.contains("voted-up") || btn.classList.contains("voted-down")) return;
        btn.classList.add(rating > 0 ? "voted-up" : "voted-down");
        other.disabled = true;
        other.style.opacity = "0.4";
        try {
            await fetch(FEEDBACK_URL, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({
                    session_id: currentSessionId,
                    trace_id:   currentTraceId,
                    rating,
                    query,
                }),
            });
        } catch { /* fire-and-forget */ }
    }

    upBtn.addEventListener("click",   () => vote(upBtn,   downBtn,  1));
    downBtn.addEventListener("click", () => vote(downBtn, upBtn,   -1));

    row.appendChild(upBtn);
    row.appendChild(downBtn);
    botMessageDiv.appendChild(row);
}

function makeFeedbackBtn(icon, label, rating) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "feedback-btn";
    btn.title = label;
    btn.innerHTML = `<span class="material-symbols-outlined">${icon}</span>${label}`;
    return btn;
}

/* ══════════════════════════════════════════════════════════════
   QUERY SUGGESTIONS
═══════════════════════════════════════════════════════════════ */
userInput.addEventListener("input", () => {
    autoResize();
    const q = userInput.value.trim();
    clearTimeout(suggestDebounce);
    if (q.length < 2) { hideSuggestions(); return; }
    suggestDebounce = setTimeout(() => fetchSuggestions(q), 320);
});

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); return; }

    if (!suggestionsDropdown.hidden && currentSuggestions.length) {
        if (e.key === "ArrowDown") {
            e.preventDefault();
            activeSuggestionIndex = Math.min(activeSuggestionIndex + 1, currentSuggestions.length - 1);
            highlightSuggestion(activeSuggestionIndex);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            activeSuggestionIndex = Math.max(activeSuggestionIndex - 1, -1);
            highlightSuggestion(activeSuggestionIndex);
        } else if (e.key === "Tab" || e.key === "Escape") {
            e.preventDefault();
            if (e.key === "Tab" && activeSuggestionIndex >= 0) {
                applySuggestion(currentSuggestions[activeSuggestionIndex].text);
            } else {
                hideSuggestions();
            }
        }
    }
});

async function fetchSuggestions(q) {
    try {
        const res  = await fetch(`${SUGGESTIONS_URL}?q=${encodeURIComponent(q)}&limit=5`);
        const data = await res.json();
        const items = data.suggestions || [];
        if (!items.length) { hideSuggestions(); return; }
        showSuggestions(items);
    } catch { hideSuggestions(); }
}

function showSuggestions(items) {
    currentSuggestions = items;
    activeSuggestionIndex = -1;
    suggestionsDropdown.innerHTML = "";

    items.forEach((item, idx) => {
        const div = document.createElement("div");
        div.className = "suggestion-item";
        div.dataset.idx = idx;
        div.innerHTML = `<span class="material-symbols-outlined">search</span><span>${escapeHtml(item.text)}</span>`;
        div.addEventListener("mousedown", (e) => { e.preventDefault(); applySuggestion(item.text); });
        suggestionsDropdown.appendChild(div);
    });

    suggestionsDropdown.hidden = false;
}

function highlightSuggestion(idx) {
    Array.from(suggestionsDropdown.children).forEach((el, i) => {
        el.classList.toggle("is-focused", i === idx);
    });
}

function applySuggestion(text) {
    userInput.value = text;
    autoResize();
    hideSuggestions();
    userInput.focus();
}

function hideSuggestions() {
    suggestionsDropdown.hidden = true;
    currentSuggestions = [];
    activeSuggestionIndex = -1;
}

document.addEventListener("click", (e) => {
    if (!suggestionsDropdown.hidden && !suggestionsDropdown.contains(e.target) && e.target !== userInput) {
        hideSuggestions();
    }
});

/* ══════════════════════════════════════════════════════════════
   DOCUMENT UPLOAD
═══════════════════════════════════════════════════════════════ */
uploadButton.addEventListener("click", () => { uploadModal.hidden = false; clearUploadStatus(); });
uploadModalClose.addEventListener("click", () => { uploadModal.hidden = true; clearUploadStatus(); });
uploadModal.addEventListener("click", (e) => { if (e.target === uploadModal) { uploadModal.hidden = true; clearUploadStatus(); } });

uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover",  (e) => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", ()  => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop",      (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    fileInput.value = "";
});

async function uploadFile(file) {
    const allowed = [".pdf", ".txt"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
        showUploadStatus("Sadece .pdf ve .txt dosyaları desteklenir.", "error");
        return;
    }

    showUploadStatus("Yükleniyor ve indeksleniyor…", "loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(UPLOAD_URL, { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showUploadStatus(`Hata: ${data.error || res.statusText}`, "error");
            return;
        }

        showUploadStatus(
            `✓ "${escapeHtml(data.filename)}" başarıyla yüklendi — ${data.chunks_added} chunk indekslendi.`,
            "success",
        );
        logEvent("upload", `${data.filename} → ${data.chunks_added} chunk eklendi.`);
    } catch (err) {
        showUploadStatus(`Yükleme başarısız: ${err.message}`, "error");
    }
}

function showUploadStatus(msg, type) {
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.textContent = msg;
    uploadStatus.hidden = false;
}

function clearUploadStatus() {
    uploadStatus.hidden = true;
    uploadStatus.textContent = "";
}

/* ══════════════════════════════════════════════════════════════
   SSE READER
═══════════════════════════════════════════════════════════════ */
async function* readSseEvents(stream) {
    const reader  = stream.getReader();
    const decoder = new TextDecoder();
    let buffer    = "";

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
    if (buffer.trim()) { const p = parseSseFrame(buffer); if (p) yield p; }
}

function parseSseFrame(frame) {
    let event = "message";
    const dataLines = [];

    frame.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:"))  dataLines.push(line.slice(5).trimStart());
    });

    if (!dataLines.length) return null;
    return { event, data: dataLines.join("\n") };
}

/* ══════════════════════════════════════════════════════════════
   AGENT FLOW UI
═══════════════════════════════════════════════════════════════ */
function markAgentActive(agent) {
    const normalized  = AGENT_ORDER.includes(agent) ? agent : "supervisor";
    const activeIndex = AGENT_ORDER.indexOf(normalized);
    visitedAgents.add(normalized);

    agentCards.forEach((card) => {
        const a = card.dataset.agent;
        const i = AGENT_ORDER.indexOf(a);
        card.classList.remove("is-active", "has-error");
        card.classList.toggle("is-complete", i < activeIndex || (visitedAgents.has(a) && a !== normalized));
        card.classList.toggle("is-active", a === normalized);
    });

    connectors.forEach((c, i) => c.classList.toggle("is-active", i < activeIndex));

    const statusNode = document.getElementById(`${normalized}Status`);
    if (statusNode) statusNode.textContent = AGENT_MESSAGES[normalized];

    activeAgentMetric.textContent = String(visitedAgents.size);
    currentAgentLabel.textContent = AGENT_LABELS[normalized];
    setStatus(`${AGENT_LABELS[normalized]} working`, "running");
    updateProgress(AGENT_PROGRESS[normalized], AGENT_MESSAGES[normalized], normalized);
    logEvent(normalized, AGENT_MESSAGES[normalized]);
}

function markComplete() {
    agentCards.forEach((c) => { c.classList.remove("is-active", "has-error"); c.classList.add("is-complete"); });
    connectors.forEach((c) => c.classList.remove("is-active"));
    updateProgress(100, "Cevap hazır", "Complete");
    currentAgentLabel.textContent = "Tamamlandı";
    setStatus("Complete", "complete");
    logEvent("done", "Akış tamamlandı, yanıt teslim edildi.");
}

function markError(message) {
    agentCards.forEach((c) => c.classList.remove("is-active"));
    const lastAgent = Array.from(visitedAgents).pop();
    const errCard = lastAgent ? document.querySelector(`[data-agent="${lastAgent}"]`) : null;
    if (errCard) errCard.classList.add("has-error");
    updateProgress(100, "Hata oluştu", "Error");
    currentAgentLabel.textContent = "Hata";
    setStatus("Error", "error");
    logEvent("error", message || "Akış hata ile sonlandı.");
}

function resetRunState() {
    visitedAgents = new Set();
    agentCards.forEach((c) => c.classList.remove("is-active", "is-complete", "has-error"));
    connectors.forEach((c) => c.classList.remove("is-active"));
    AGENT_ORDER.forEach((a) => {
        const n = document.getElementById(`${a}Status`);
        if (n) n.textContent = AGENT_MESSAGES[a];
    });
    retrievalList.innerHTML = `
        <div class="empty-state">
            <span class="material-symbols-outlined">database</span>
            <p>Yeni bir sorgu çalıştığında kanıtlar burada listelenir.</p>
        </div>`;
    retrievalLabel.textContent = "Boş";
    activeAgentMetric.textContent = "0";
    documentMetric.textContent    = "0";
    elapsedMetric.textContent     = "0.0s";
}

/* ══════════════════════════════════════════════════════════════
   RETRIEVAL RESULTS
═══════════════════════════════════════════════════════════════ */
function showSearchResults(results) {
    retrievalList.innerHTML = "";
    documentMetric.textContent = String(results.length);
    retrievalLabel.textContent = `${results.length} doküman`;

    if (!results.length) {
        retrievalList.innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">travel_explore</span>
                <p>Bu sorgu için doküman bulunamadı.</p>
            </div>`;
        return;
    }

    results.forEach((result, index) => {
        const card       = document.createElement("article");
        card.className   = `retrieval-card ${result.relevant ? "relevant" : "not-relevant"}`;
        const score      = typeof result.score === "number"       ? result.score.toFixed(4)       : "n/a";
        const denseScore = typeof result.dense_score === "number" ? result.dense_score.toFixed(3) : "0.000";
        const bm25Score  = typeof result.bm25_score === "number"  ? result.bm25_score.toFixed(3)  : "0.000";

        card.innerHTML = `
            <div class="retrieval-head">
                <strong>${escapeHtml(result.title || `Doc ${index + 1}`)}</strong>
                <span class="score-pill">score ${score}</span>
                <span class="relevance-pill">dense ${denseScore}</span>
                <span class="relevance-pill">bm25 ${bm25Score}</span>
            </div>
            <small>${escapeHtml(result.source || "unknown")}</small>
            <p>${escapeHtml(result.content || "")}</p>`;
        retrievalList.appendChild(card);
    });
}

/* ══════════════════════════════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════════════════════════════ */
function addMessage(text, type) {
    const div     = document.createElement("div");
    div.className = `message ${type}-message`;

    const avatar       = document.createElement("div");
    avatar.className   = "message-avatar";
    avatar.textContent = type === "user" ? "YOU" : "AI";

    const content       = document.createElement("div");
    content.className   = "message-content";
    content.textContent = text;

    div.appendChild(avatar);
    div.appendChild(content);
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function createBotMessage() {
    const div     = document.createElement("div");
    div.className = "message bot-message";

    const avatar       = document.createElement("div");
    avatar.className   = "message-avatar";
    avatar.textContent = "AI";

    const content     = document.createElement("div");
    content.className = "message-content";

    div.appendChild(avatar);
    div.appendChild(content);
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function addTypingIndicator() {
    const div     = document.createElement("div");
    div.className = "message bot-message";
    div.id        = "typingIndicator";

    const avatar       = document.createElement("div");
    avatar.className   = "message-avatar";
    avatar.textContent = "AI";

    const content     = document.createElement("div");
    content.className = "message-content typing-indicator";
    content.innerHTML = "<span></span><span></span><span></span>";

    div.appendChild(avatar);
    div.appendChild(content);
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

/* ══════════════════════════════════════════════════════════════
   UI HELPERS
═══════════════════════════════════════════════════════════════ */
function updateProgress(percent, label, stateText) {
    const p = Math.max(0, Math.min(100, percent));
    progressBar.style.width        = `${p}%`;
    progressPercent.textContent    = `${p}%`;
    progressLabel.textContent      = label;
    flowStateText.textContent      = stateText;
    if (heroMeter) heroMeter.style.setProperty("--meter-progress", `${p}%`);
}

function setStatus(label, mode) {
    statusBadge.className = `status-pill ${mode || ""}`.trim();
    statusBadge.innerHTML = `<span class="status-dot"></span>${label}`;
}

function logEvent(type, message) {
    if (!eventLog) return;
    const item = document.createElement("div");
    item.className = `event-item ${type === "error" ? "error" : type === "done" ? "done" : ""}`;
    const time = new Date().toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    item.innerHTML = `<span>${time} | ${escapeHtml(type)}</span><p>${escapeHtml(message)}</p>`;
    eventLog.appendChild(item);
    eventLog.scrollTop = eventLog.scrollHeight;
}

function startElapsedTimer() {
    stopElapsedTimer();
    startedAt    = performance.now();
    elapsedTimer = window.setInterval(() => {
        elapsedMetric.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
    }, 100);
}

function stopElapsedTimer() {
    if (elapsedTimer) { window.clearInterval(elapsedTimer); elapsedTimer = null; }
}

function setControlsDisabled(disabled) {
    sendButton.disabled = disabled;
    userInput.disabled  = disabled;
}

function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = `${userInput.scrollHeight}px`;
}

function safeJson(value) {
    try { return JSON.parse(value); } catch { return null; }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function generateUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
    });
}

/* ══════════════════════════════════════════════════════════════
   WELCOME SUGGESTIONS + NEW SESSION
═══════════════════════════════════════════════════════════════ */
const WELCOME_HTML = `
    <div class="welcome">
        <p class="welcome-eyebrow">Merhaba 👋</p>
        <h1 class="welcome-title">Bugün neyi araştıralım?</h1>
        <p class="welcome-lead">
            Bir soru yazın. Supervisor yönlendirir, Researcher kanıt toplar,
            Writer yanıtı kurar, Reviewer kalite kontrol yapar — her adımı
            sağ tarafta canlı izleyebilirsiniz.
        </p>
        <div class="suggestion-row">
            <button class="suggestion" type="button">Belgelerdeki ana bulguları özetle</button>
            <button class="suggestion" type="button">İki kaynağı karşılaştır</button>
            <button class="suggestion" type="button">Kısa bir kronoloji çıkar</button>
        </div>
    </div>`;

function bindSuggestions() {
    chatMessages.querySelectorAll(".suggestion").forEach((chip) => {
        chip.addEventListener("click", () => {
            userInput.value = chip.textContent.trim();
            autoResize();
            sendMessage();
        });
    });
}

newSessionButton.addEventListener("click", () => {
    currentSessionId = generateUUID();
    localStorage.setItem("rag_session_id", currentSessionId);
    currentTraceId   = "";
    chatMessages.innerHTML = WELCOME_HTML;
    bindSuggestions();
    eventLog.innerHTML = "";
    resetRunState();
    updateProgress(0, "Hazır", "Idle");
    setStatus("Hazır", "");
    logEvent("idle", "Yeni oturum başlatıldı.");
    userInput.focus();
});

chatForm.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });

window.addEventListener("load", () => {
    bindSuggestions();
    scrollToBottom();
    updateProgress(0, "Hazır", "Idle");
    logEvent("idle", "Sistem hazır. Yeni bir istek bekleniyor.");
});
