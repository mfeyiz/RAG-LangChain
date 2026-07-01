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
const docList           = document.getElementById("docList");
const docCountTag       = document.getElementById("docCountTag");
const chatView          = document.getElementById("chatView");
const libraryView       = document.getElementById("libraryView");
const tabChat           = document.getElementById("tabChat");
const tabLibrary        = document.getElementById("tabLibrary");
const libraryList       = document.getElementById("libraryList");
const libraryCountTag   = document.getElementById("libraryCountTag");
const libraryViewTabs   = document.getElementById("libraryViewTabs");
const libraryDocTitle   = document.getElementById("libraryDocTitle");
const libraryDocBody    = document.getElementById("libraryDocBody");
const libraryDownloads  = document.getElementById("libraryDownloads");
const attachButton      = document.getElementById("attachButton");
const imageInput        = document.getElementById("imageInput");
const attachPreview     = document.getElementById("attachPreview");
const suggestionsDropdown = document.getElementById("suggestionsDropdown");
const statusBanner      = document.getElementById("statusBanner");
const statusBannerText  = document.getElementById("statusBannerText");
const heroMeter         = document.querySelector(".hero-meter");
const agentCards        = Array.from(document.querySelectorAll(".agent-card"));
const connectors        = Array.from(document.querySelectorAll(".flow-connector"));

/* ── Constants ───────────────────────────────────────────────── */
const API_URL          = "/ask";
const SUGGESTIONS_URL  = "/suggestions";
const FEEDBACK_URL     = "/feedback";
const UPLOAD_URL       = "/upload";
const STATUS_URL       = "/status";
const ADMIN_DOCS_URL   = "/admin/documents";
const DOC_CONTENT_URL  = "/documents";

const AGENT_ORDER    = ["supervisor", "researcher", "writer", "reviewer", "editor"];
const AGENT_PROGRESS = { supervisor: 18, researcher: 45, writer: 74, reviewer: 92, editor: 74 };
const AGENT_LABELS   = { supervisor: "Supervisor", researcher: "Researcher", writer: "Writer", reviewer: "Reviewer", editor: "Editor" };
const AGENT_MESSAGES = {
    supervisor: "Yönlendirme kararı veriliyor",
    researcher: "Vektör veritabanı taranıyor",
    writer:     "Kanıtlara dayalı yanıt yazılıyor",
    reviewer:   "Cevap kalite kontrolden geçiyor",
    editor:     "Bilgi güncelleniyor ve yeniden indeksleniyor",
};

/* ── Auth state ──────────────────────────────────────────────── */
const AUTH_LOGIN_URL = "/auth/login";
let authToken = localStorage.getItem("rag_auth_token") || "";
let authUser  = localStorage.getItem("rag_auth_user")  || "";

function isAuthenticated() { return Boolean(authToken); }

/* Merge the bearer token into request headers when logged in. */
function authHeaders(extra = {}) {
    const headers = { ...extra };
    if (authToken) headers.Authorization = `Bearer ${authToken}`;
    return headers;
}

function setAuth(token, username) {
    authToken = token || "";
    authUser  = username || "";
    if (authToken) {
        localStorage.setItem("rag_auth_token", authToken);
        localStorage.setItem("rag_auth_user", authUser);
    } else {
        localStorage.removeItem("rag_auth_token");
        localStorage.removeItem("rag_auth_user");
    }
    renderAuthState();
}

/* ── Session state ───────────────────────────────────────────── */
let currentSessionId = localStorage.getItem("rag_session_id") || generateUUID();
localStorage.setItem("rag_session_id", currentSessionId);
let currentTraceId = "";

let elapsedTimer  = null;
let startedAt     = 0;
let visitedAgents = new Set();

/* ── Multimodal attachments (data URLs sent with the next query) ── */
let pendingImages = [];

function renderAttachPreview() {
    if (!attachPreview) return;
    attachPreview.innerHTML = "";
    attachPreview.hidden = pendingImages.length === 0;
    pendingImages.forEach((dataUrl, i) => {
        const thumb = document.createElement("div");
        thumb.className = "attach-thumb";
        const img = document.createElement("img");
        img.src = dataUrl;
        thumb.appendChild(img);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "attach-remove";
        remove.setAttribute("aria-label", "Kaldır");
        remove.innerHTML = `<span class="material-symbols-outlined">close</span>`;
        remove.addEventListener("click", () => { pendingImages.splice(i, 1); renderAttachPreview(); });
        thumb.appendChild(remove);
        attachPreview.appendChild(thumb);
    });
}

function addImageFiles(files) {
    Array.from(files || []).forEach((file) => {
        if (!file.type.startsWith("image/") || pendingImages.length >= 4) return;
        const reader = new FileReader();
        reader.onload = () => { pendingImages.push(reader.result); renderAttachPreview(); };
        reader.readAsDataURL(file);
    });
}

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
    userInput.value = "";
    autoResize();
    runQuery(message);
}

// Core query runner. `allowWeb` re-runs an approved web-search fallback;
// `echoUser=false` skips re-printing the user bubble for those re-runs.
async function runQuery(message, { allowWeb = false, echoUser = true } = {}) {
    if (!message || sendButton.disabled) return;

    hideSuggestions();
    resetRunState();
    const welcome = chatMessages.querySelector(".welcome");
    if (welcome) welcome.remove();
    // Attach images only on the first run of a query (not on web-approval re-runs).
    const outgoingImages = (echoUser && !allowWeb) ? pendingImages.slice() : [];
    if (echoUser) addMessage(message, "user", outgoingImages);
    if (outgoingImages.length) { pendingImages = []; renderAttachPreview(); }
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
    let webSources        = null;   // web result list, used to linkify citations
    let awaitingApproval  = false;  // showing the "search the web?" prompt
    let awaitingEditApproval = false; // showing @update diff awaiting Approve/Reject
    let searchMeta        = null;  // search_results for the current answer (citation panel)

    try {
        const response = await fetch(API_URL, {
            method:  "POST",
            headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
            body:    JSON.stringify({ query: message, session_id: currentSessionId, allow_web: allowWeb, images: outgoingImages }),
        });

        if (response.status === 403) {
            // @update attempted without auth — prompt the user to sign in.
            typingIndicator.remove();
            let detail = "@update için giriş yapmalısınız.";
            try { const p = await response.json(); if (p?.error) detail = p.error; } catch {}
            throw new Error(detail, { cause: "auth" });
        }

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
                searchMeta = results;
                botMessageDiv.__searchMeta = results;
                const webResults = results.filter((r) => r.origin === "web");
                if (webResults.length) {
                    webSources = webResults;
                    logEvent("retrieval", `İnternetten ${webResults.length} kaynak getirildi.`);
                } else {
                    logEvent("retrieval", `${results.length} doküman skoru alındı.`);
                }
            }

            /* figures anchored to the retrieved context */
            if (event.event === "context_images") {
                const images = safeJson(event.data) || [];
                showContextImages(botMessageDiv, images);
                if (images.length) logEvent("retrieval", `${images.length} ilgili görsel bulundu.`);
            }

            /* weak RAG match — ask before searching the web */
            if (event.event === "web_search_prompt") {
                const payload = safeJson(event.data) || {};
                awaitingApproval = true;
                streamingStarted = true;
                showWebSearchPrompt(botMessageDiv, payload.query || message);
            }

            /* editor write-back result (@update) */
            if (event.event === "edit_result") {
                const payload = safeJson(event.data);
                if (payload) {
                    showEditResult(botMessageDiv, payload);
                    loadLibrary();  // reflect the new chunk count / edited file
                }
            }

            /* editor DIFF PREVIEW awaiting human approval (@update HITL) */
            if (event.event === "edit_preview") {
                const payload = safeJson(event.data);
                if (payload) {
                    streamingStarted = true;
                    awaitingEditApproval = true;
                    showEditPreview(botMessageDiv, payload);
                }
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
                renderAnswer(contentDiv, fullText, webSources, searchMeta);
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

        if (!fullText.trim() && !awaitingApproval && !awaitingEditApproval) {
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
        if (error.cause === "auth") openAuthModal();
    } finally {
        stopElapsedTimer();
        setControlsDisabled(false);
        userInput.focus();
    }
}

/* ══════════════════════════════════════════════════════════════
   EDITOR WRITE-BACK RESULT (@update)
═══════════════════════════════════════════════════════════════ */
function showEditResult(botMessageDiv, payload) {
    const card = document.createElement("div");
    card.className = "edit-result";

    if (payload.file) {
        card.classList.add("is-clickable");
        card.title = "Güncellenen belgeyi görüntüle";
        card.addEventListener("click", (e) => {
            if (e.target.closest("a")) return;  // let the download link work normally
            openUpdatedDoc(payload.file);
        });
    }

    const title = document.createElement("div");
    title.className = "edit-result-title";
    title.innerHTML = `<span class="material-symbols-outlined">drive_file_rename_outline</span> ${payload.file || "Çalışma alanı güncellendi"}`;
    card.appendChild(title);

    if (payload.summary) {
        const summary = document.createElement("p");
        summary.className = "edit-result-summary";
        summary.textContent = payload.summary;
        card.appendChild(summary);
    }

    if (payload.pdf_url) {
        const link = document.createElement("a");
        link.className = "edit-result-download";
        link.href = payload.pdf_url;
        link.target = "_blank";
        link.rel = "noopener";
        link.innerHTML = `<span class="material-symbols-outlined">download</span> Güncel PDF'i indir`;
        card.appendChild(link);
    }

    if (payload.file) {
        const hint = document.createElement("div");
        hint.className = "edit-result-hint";
        hint.innerHTML = `<span class="material-symbols-outlined">open_in_new</span> Değişikliği görmek için tıklayın`;
        card.appendChild(hint);
    }

    botMessageDiv.appendChild(card);
    logEvent("editor", `Çalışma alanı güncellendi: ${payload.file || ""}`);
    scrollToBottom();
}

// Jump from an @update result card straight to the edited document, showing the
// originals↔workspace diff and scrolling to the first changed line so the user
// can immediately see (and confirm) what was written.
async function openUpdatedDoc(source) {
    switchTab("library");
    await selectDoc(source);
    renderDocView("compare");
    requestAnimationFrame(() => {
        const firstChange = libraryDocBody.querySelector(".diff-line.added, .diff-line.removed");
        if (firstChange) firstChange.scrollIntoView({ behavior: "smooth", block: "center" });
    });
}

/* ══════════════════════════════════════════════════════════════
   EDIT PREVIEW (diff viewer) + Human-in-the-loop approval (@update)
═══════════════════════════════════════════════════════════════ */
function showEditPreview(botMessageDiv, payload) {
    const card = document.createElement("div");
    card.className = "edit-preview";

    const header = document.createElement("div");
    header.className = "edit-preview-head";
    header.innerHTML = `
        <span class="material-symbols-outlined">rule</span>
        <span class="edit-preview-title">Değişiklik önizleme</span>
        <span class="edit-preview-file">${escapeHtml(payload.file || "")}</span>`;
    card.appendChild(header);

    if (payload.instruction) {
        const instr = document.createElement("p");
        instr.className = "edit-preview-instruction";
        instr.innerHTML = `<strong>Talimat:</strong> ${escapeHtml(payload.instruction)}`;
        card.appendChild(instr);
    }

    const diff = payload.diff || [];
    if (!diff.length) {
        const note = document.createElement("p");
        note.className = "edit-preview-empty";
        note.textContent = "Değişiklik tespit edilmedi.";
        card.appendChild(note);
        botMessageDiv.appendChild(card);
        return;
    }

    const viewer = document.createElement("div");
    viewer.className = "diff-viewer";
    diff.forEach((row) => {
        const line = document.createElement("div");
        const type = row.type || "same";
        line.className = `diff-viewer-line diff-${type}`;
        const sign = type === "added" ? "+" : type === "removed" ? "-" : type === "ellipsis" ? "…" : " ";
        const txt = type === "removed" ? row.before : type === "added" ? row.after : row.after ?? row.before;
        if (type === "ellipsis") {
            line.textContent = "…";
        } else {
            line.innerHTML = `<span class="diff-gutter">${sign}</span><span class="diff-content">${escapeHtml(txt ?? "") || "&nbsp;"}</span>`;
        }
        viewer.appendChild(line);
    });
    card.appendChild(viewer);

    const meta = document.createElement("div");
    meta.className = "edit-preview-meta";
    const added = diff.filter((r) => r.type === "added").length;
    const removed = diff.filter((r) => r.type === "removed").length;
    meta.textContent = `+${added} satır · −${removed} satır`;
    card.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "edit-preview-actions";

    const approveBtn = document.createElement("button");
    approveBtn.type = "button";
    approveBtn.className = "primary-button edit-approve";
    approveBtn.innerHTML = `<span class="material-symbols-outlined">check</span> Onayla`;
    approveBtn.disabled = false;

    const rejectBtn = document.createElement("button");
    rejectBtn.type = "button";
    rejectBtn.className = "ghost-button edit-reject";
    rejectBtn.innerHTML = `<span class="material-symbols-outlined">close</span> Reddet`;

    actions.appendChild(rejectBtn);
    actions.appendChild(approveBtn);
    card.appendChild(actions);

    const status = document.createElement("div");
    status.className = "edit-preview-status";
    card.appendChild(status);

    botMessageDiv.appendChild(card);
    logEvent("editor", `Değişiklik önizlemesi hazır — onay bekleniyor (${payload.file || ""}).`);
    scrollToBottom();

    async function approve() {
        approveBtn.disabled = true; rejectBtn.disabled = true;
        status.textContent = "Uygulanıyor…"; status.className = "edit-preview-status is-pending";
        try {
            const res = await fetch("/update/apply", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ token: payload.token }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Onay başarısız.");
            status.innerHTML = `<span class="material-symbols-outlined">check_circle</span> ${escapeHtml(data.reply || "Değişiklik uygulandı.")}`;
            status.className = "edit-preview-status is-ok";
            card.querySelector(".diff-viewer").classList.add("diff-applied");
            card.querySelector(".edit-preview-actions").remove();
            addMessage(data.reply, "assistant");
            loadLibrary();
        } catch (err) {
            status.textContent = `Hata: ${err.message}`;
            status.className = "edit-preview-status is-error";
            approveBtn.disabled = false; rejectBtn.disabled = false;
        }
    }

    async function reject() {
        approveBtn.disabled = true; rejectBtn.disabled = true;
        try {
            await fetch("/update/reject", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ token: payload.token }),
            });
        } catch { /* fire-and-forget */ }
        status.innerHTML = `<span class="material-symbols-outlined">cancel</span> Değişiklik reddedildi — belgeye işlenmedi.`;
        status.className = "edit-preview-status is-rejected";
        card.querySelector(".edit-preview-actions").remove();
        logEvent("editor", "Değişiklik kullanıcı tarafından reddedildi.");
    }

    approveBtn.addEventListener("click", approve);
    rejectBtn.addEventListener("click", reject);
}

/* ══════════════════════════════════════════════════════════════
   FEEDBACK
═══════════════════════════════════════════════════════════════ */
function addFeedbackRow(botMessageDiv, query) {
    const wrap = document.createElement("div");
    wrap.className = "feedback-wrap";

    const row = document.createElement("div");
    row.className = "feedback-row";

    const label   = document.createElement("span");
    label.className = "feedback-label";
    label.textContent = "Bu yanıt yardımcı oldu mu?";

    const upBtn   = makeFeedbackBtn("thumb_up",   "İyi yanıt",  1);
    const downBtn = makeFeedbackBtn("thumb_down", "Kötü yanıt", -1);

    let submitted = false;

    async function sendFeedback(rating, comment) {
        try {
            await fetch(FEEDBACK_URL, {
                method:  "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body:    JSON.stringify({
                    session_id: currentSessionId,
                    trace_id:   currentTraceId,
                    rating,
                    query,
                    comment: comment || "",
                }),
            });
        } catch { /* fire-and-forget */ }
    }

    function lockButtons(rating) {
        upBtn.disabled = downBtn.disabled = true;
        (rating > 0 ? upBtn : downBtn).classList.add(rating > 0 ? "voted-up" : "voted-down");
        (rating > 0 ? downBtn : upBtn).style.opacity = "0.35";
    }

    upBtn.addEventListener("click", () => {
        if (submitted) return;
        submitted = true;
        lockButtons(1);
        label.textContent = "Teşekkürler! 👍";
        sendFeedback(1, "");
    });

    downBtn.addEventListener("click", () => {
        if (submitted) return;
        submitted = true;
        lockButtons(-1);
        label.textContent = "Geri bildiriminiz için teşekkürler.";
        // Capture the rating immediately, then let the user enrich it with a comment.
        sendFeedback(-1, "");
        showCommentBox(wrap, query, sendComment);
    });

    async function sendComment(comment) {
        if (!comment) return;
        try {
            await fetch(`${FEEDBACK_URL}/comment`, {
                method:  "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body:    JSON.stringify({
                    session_id: currentSessionId,
                    trace_id:   currentTraceId,
                    comment,
                }),
            });
        } catch { /* fire-and-forget */ }
    }

    row.appendChild(label);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    wrap.appendChild(row);
    botMessageDiv.appendChild(wrap);
}

function showCommentBox(wrap, query, onSubmit) {
    const box = document.createElement("div");
    box.className = "feedback-comment";

    const textarea = document.createElement("textarea");
    textarea.placeholder = "Neyi iyileştirebiliriz? (isteğe bağlı)";
    textarea.rows = 2;

    const submit = document.createElement("button");
    submit.type = "button";
    submit.className = "feedback-comment-submit";
    submit.textContent = "Gönder";

    let sent = false;
    function finish() {
        if (sent) return;
        sent = true;
        onSubmit(textarea.value.trim());
        box.innerHTML = `<span class="feedback-comment-done">Yorumunuz kaydedildi.</span>`;
    }

    submit.addEventListener("click", finish);
    box.appendChild(textarea);
    box.appendChild(submit);
    wrap.appendChild(box);
    textarea.focus();
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

/* Image attachments: button, file picker, and clipboard paste. */
if (attachButton) attachButton.addEventListener("click", () => imageInput?.click());
if (imageInput) imageInput.addEventListener("change", () => { addImageFiles(imageInput.files); imageInput.value = ""; });
userInput.addEventListener("paste", (e) => {
    const items = Array.from(e.clipboardData?.items || []).filter((it) => it.type.startsWith("image/"));
    if (!items.length) return;
    e.preventDefault();
    addImageFiles(items.map((it) => it.getAsFile()).filter(Boolean));
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
uploadButton.addEventListener("click", () => { uploadModal.hidden = false; clearUploadStatus(); loadDocumentList(); });
uploadModalClose.addEventListener("click", () => { uploadModal.hidden = true; clearUploadStatus(); });
uploadModal.addEventListener("click", (e) => { if (e.target === uploadModal) { uploadModal.hidden = true; clearUploadStatus(); } });


/* ══════════════════════════════════════════════════════════════
   AUTHENTICATION (login modal + token state)
═══════════════════════════════════════════════════════════════ */
const authButton      = document.getElementById("authButton");
const authButtonLabel = document.getElementById("authButtonLabel");
const authModal       = document.getElementById("authModal");
const authModalClose  = document.getElementById("authModalClose");
const authForm        = document.getElementById("authForm");
const authUsername    = document.getElementById("authUsername");
const authPassword    = document.getElementById("authPassword");
const authError       = document.getElementById("authError");

function renderAuthState() {
    if (!authButtonLabel) return;
    authButtonLabel.textContent = isAuthenticated() ? (authUser || "Çıkış yap") : "Giriş yap";
    if (authButton) authButton.title = isAuthenticated()
        ? `${authUser} olarak giriş yapıldı — çıkış için tıklayın`
        : "Giriş yap";
}

function openAuthModal() {
    if (!authModal) return;
    authError.hidden = true;
    authForm.reset();
    authModal.hidden = false;
    authUsername.focus();
}

function closeAuthModal() { if (authModal) authModal.hidden = true; }

if (authButton) {
    authButton.addEventListener("click", () => {
        if (isAuthenticated()) {
            // Toggle to logout when already signed in.
            setAuth("", "");
            logEvent("auth", "Oturum kapatıldı.");
        } else {
            openAuthModal();
        }
    });
}
if (authModalClose) authModalClose.addEventListener("click", closeAuthModal);
if (authModal) authModal.addEventListener("click", (e) => { if (e.target === authModal) closeAuthModal(); });

if (authForm) {
    authForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        authError.hidden = true;
        const username = authUsername.value.trim();
        const password = authPassword.value;
        if (!username || !password) return;

        try {
            const res = await fetch(AUTH_LOGIN_URL, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ username, password }),
            });
            const data = await res.json();
            if (!res.ok) {
                authError.textContent = data.error || "Giriş başarısız oldu.";
                authError.hidden = false;
                return;
            }
            setAuth(data.token, data.username || username);
            closeAuthModal();
            logEvent("auth", `${data.username || username} olarak giriş yapıldı.`);
        } catch (err) {
            authError.textContent = "Sunucuya ulaşılamadı.";
            authError.hidden = false;
        }
    });
}

renderAuthState();

tabChat.addEventListener("click", () => switchTab("chat"));
tabLibrary.addEventListener("click", () => switchTab("library"));
libraryViewTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".lib-vtab");
    if (btn) renderDocView(btn.dataset.view);
});

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
    const allowed = [".pdf", ".docx"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
        showUploadStatus("Sadece .pdf ve .docx dosyaları desteklenir.", "error");
        return;
    }

    showUploadStatus("Yükleniyor ve indeksleniyor…", "loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(UPLOAD_URL, { method: "POST", headers: authHeaders(), body: formData });
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
        loadDocumentList();
        loadLibrary();
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
   DOCUMENT LIST (admin: list + delete)
═══════════════════════════════════════════════════════════════ */
async function loadDocumentList() {
    docCountTag.textContent = "Yükleniyor…";
    docList.innerHTML = "";

    try {
        const res  = await fetch(ADMIN_DOCS_URL, { headers: authHeaders() });
        const data = await res.json();
        const docs = data.documents || [];

        docCountTag.textContent = `${docs.length} kaynak`;

        if (!docs.length) {
            docList.innerHTML = `
                <div class="doc-list-empty">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>Henüz belge yok.</p>
                </div>`;
            return;
        }

        docs.forEach((doc) => {
            const row = document.createElement("div");
            row.className = "doc-row";
            row.dataset.source = doc.source;

            const info = document.createElement("div");
            info.className = "doc-row-info";
            info.innerHTML = `
                <span class="doc-row-name">${escapeHtml(doc.source.split("/").pop())}</span>
                <span class="doc-row-meta">${doc.chunks} chunk · <small>${escapeHtml(doc.source)}</small></span>`;

            const delBtn = document.createElement("button");
            delBtn.type = "button";
            delBtn.className = "doc-delete-btn";
            delBtn.title = "Sil";
            delBtn.innerHTML = `<span class="material-symbols-outlined">delete</span>`;
            delBtn.addEventListener("click", () => deleteDocument(doc.source, row));

            row.appendChild(info);
            row.appendChild(delBtn);
            docList.appendChild(row);
        });
    } catch {
        docCountTag.textContent = "Hata";
        docList.innerHTML = `<div class="doc-list-empty"><p>Belgeler yüklenemedi.</p></div>`;
    }
}

async function deleteDocument(source, rowEl) {
    if (!confirm(`"${source.split("/").pop()}" belgesini silmek istediğinizden emin misiniz?`)) return;

    rowEl.classList.add("is-deleting");
    try {
        const res  = await fetch(`${ADMIN_DOCS_URL}/${encodeURIComponent(source)}`, { method: "DELETE", headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
            alert(data.error || "Silme başarısız oldu.");
            rowEl.classList.remove("is-deleting");
            return;
        }
        rowEl.remove();
        logEvent("upload", `${source.split("/").pop()} silindi (${data.chunks_deleted} chunk).`);
        const remaining = docList.querySelectorAll(".doc-row").length;
        docCountTag.textContent = `${remaining} kaynak`;
        if (!remaining) {
            docList.innerHTML = `
                <div class="doc-list-empty">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>Henüz belge yok.</p>
                </div>`;
        }
        loadLibrary();
    } catch {
        alert("Silme isteği başarısız oldu.");
        rowEl.classList.remove("is-deleting");
    }
}

/* ══════════════════════════════════════════════════════════════
   LIBRARY TAB (view + compare originals vs workspace)
═══════════════════════════════════════════════════════════════ */
let libraryDoc = null;   // { source, original, workspace, original_url, workspace_pdf_url }
let libraryView_ = "workspace";

function switchTab(tab) {
    const isLibrary = tab === "library";
    chatView.hidden = isLibrary;
    libraryView.hidden = !isLibrary;
    tabChat.classList.toggle("is-active", !isLibrary);
    tabLibrary.classList.toggle("is-active", isLibrary);
    if (isLibrary) loadLibrary();
}

async function loadLibrary() {
    try {
        const res  = await fetch(`${ADMIN_DOCS_URL}?channel=workspace`, { headers: authHeaders() });
        const data = await res.json();
        const docs = data.documents || [];

        libraryCountTag.textContent = String(docs.length);
        if (!docs.length) {
            libraryList.innerHTML = `
                <div class="empty-state">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>Henüz belge yok. "Belge yükle" ile PDF/DOCX ekleyin.</p>
                </div>`;
            return;
        }

        libraryList.innerHTML = "";
        docs.forEach((doc) => {
            const row = document.createElement("button");
            row.type = "button";
            row.className = "library-row";
            row.dataset.source = doc.source;
            if (libraryDoc && libraryDoc.source === doc.source) row.classList.add("is-active");
            row.innerHTML = `
                <span class="material-symbols-outlined">description</span>
                <span class="library-row-text">
                    <span class="library-row-name">${escapeHtml(doc.source)}</span>
                    <span class="library-row-meta">${doc.chunks} chunk</span>
                </span>`;
            row.addEventListener("click", () => selectDoc(doc.source));
            libraryList.appendChild(row);
        });
    } catch {
        libraryCountTag.textContent = "!";
        libraryList.innerHTML = `<div class="empty-state"><p>Belgeler yüklenemedi.</p></div>`;
    }
}

async function selectDoc(source) {
    libraryDocTitle.textContent = source;
    libraryViewTabs.hidden = false;
    libraryDownloads.innerHTML = "";
    libraryDocBody.innerHTML = `<div class="compare-loading">Yükleniyor…</div>`;
    libraryList.querySelectorAll(".library-row").forEach((r) =>
        r.classList.toggle("is-active", r.dataset.source === source));

    try {
        const [origRes, wsRes] = await Promise.all([
            fetch(`${DOC_CONTENT_URL}/${encodeURIComponent(source)}/content?channel=originals`),
            fetch(`${DOC_CONTENT_URL}/${encodeURIComponent(source)}/content?channel=workspace`),
        ]);
        const orig = origRes.ok ? await origRes.json() : { markdown: "" };
        const ws   = wsRes.ok   ? await wsRes.json()   : { markdown: "" };

        libraryDoc = {
            source,
            original: orig.markdown || "",
            workspace: ws.markdown || "",
            original_url: orig.original_url || "",
            workspace_pdf_url: ws.workspace_pdf_url || "",
        };

        let dl = "";
        if (libraryDoc.original_url)
            dl += `<a class="ghost-button" href="${libraryDoc.original_url}" target="_blank" rel="noopener"><span class="material-symbols-outlined">picture_as_pdf</span> Orijinal</a>`;
        if (libraryDoc.workspace_pdf_url)
            dl += `<a class="ghost-button" href="${libraryDoc.workspace_pdf_url}" target="_blank" rel="noopener"><span class="material-symbols-outlined">download</span> Güncel PDF</a>`;
        libraryDownloads.innerHTML = dl;

        renderDocView(libraryView_);
    } catch {
        libraryDocBody.innerHTML = `<div class="compare-loading">Belge yüklenemedi.</div>`;
    }
}

function renderDocView(view) {
    libraryView_ = view;
    libraryViewTabs.querySelectorAll(".lib-vtab").forEach((b) =>
        b.classList.toggle("is-active", b.dataset.view === view));
    if (!libraryDoc) return;

    if (view === "compare") {
        const rows = diffLines(libraryDoc.original.split("\n"), libraryDoc.workspace.split("\n"));
        let left = "", right = "";
        rows.forEach((r) => {
            const lCls = r.type === "removed" ? "removed" : (r.type === "added" ? "filler" : "same");
            const rCls = r.type === "added"   ? "added"   : (r.type === "removed" ? "filler" : "same");
            left  += `<div class="diff-line ${lCls}">${r.left  === null ? "" : escapeHtml(r.left)  || "&nbsp;"}</div>`;
            right += `<div class="diff-line ${rCls}">${r.right === null ? "" : escapeHtml(r.right) || "&nbsp;"}</div>`;
        });
        const identical = libraryDoc.original === libraryDoc.workspace;
        libraryDocBody.innerHTML = `
            ${identical ? `<div class="compare-note">Orijinal ve çalışma alanı şu an aynı (henüz düzenleme yapılmamış).</div>` : ""}
            <div class="compare-grid">
                <div class="compare-col">
                    <div class="compare-col-head">Orijinal <small>(salt-okunur)</small></div>
                    <div class="compare-pane">${left}</div>
                </div>
                <div class="compare-col">
                    <div class="compare-col-head">Çalışma Alanı <small>(düzenlenen)</small></div>
                    <div class="compare-pane">${right}</div>
                </div>
            </div>`;
        return;
    }

    const md = view === "originals" ? libraryDoc.original : libraryDoc.workspace;
    libraryDocBody.innerHTML = `<article class="md-render">${renderMarkdown(md)}</article>`;
}

/* Minimal, safe Markdown → HTML renderer (escapes first, then formats). */
function renderMarkdown(md) {
    const lines = (md || "").split("\n");
    let html = "", inCode = false, listType = null;
    const closeList = () => { if (listType) { html += `</${listType}>`; listType = null; } };
    const inline = (t) => escapeHtml(t)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        // Inline figures the backend served — restricted to same-origin /images/
        // URLs so answers can't be coaxed into loading arbitrary remote content.
        .replace(/!\[([^\]]*)\]\((\/images\/[^)\s]+)\)/g, '<img src="$2" alt="$1" loading="lazy" class="answer-image">')
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
        .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    for (const raw of lines) {
        const line = raw.replace(/\s+$/, "");
        if (line.trim().startsWith("```")) { inCode = !inCode; html += inCode ? "<pre><code>" : "</code></pre>"; continue; }
        if (inCode) { html += escapeHtml(raw) + "\n"; continue; }

        const h = /^(#{1,6})\s+(.*)$/.exec(line);
        if (h) { closeList(); const n = h[1].length; html += `<h${n}>${inline(h[2])}</h${n}>`; continue; }
        if (/^\s*[-*+]\s+/.test(line)) { if (listType !== "ul") { closeList(); listType = "ul"; html += "<ul>"; } html += `<li>${inline(line.replace(/^\s*[-*+]\s+/, ""))}</li>`; continue; }
        if (/^\s*\d+\.\s+/.test(line)) { if (listType !== "ol") { closeList(); listType = "ol"; html += "<ol>"; } html += `<li>${inline(line.replace(/^\s*\d+\.\s+/, ""))}</li>`; continue; }
        if (/^\s*>\s?/.test(line)) { closeList(); html += `<blockquote>${inline(line.replace(/^\s*>\s?/, ""))}</blockquote>`; continue; }
        if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { closeList(); html += "<hr>"; continue; }
        if (line.includes("|") && line.split("|").length > 2 && /\S/.test(line)) { closeList(); html += renderTableRow(line, inline); continue; }
        if (!line.trim()) { closeList(); continue; }
        closeList(); html += `<p>${inline(line)}</p>`;
    }
    closeList();
    if (inCode) html += "</code></pre>";
    return html.replace(/(<tr[\s\S]*?<\/tr>)(?!\s*<tr)/g, "<table>$&</table>")
               .replace(/<\/table>\s*<table>/g, "");
}

function renderTableRow(line, inline) {
    if (/^\s*\|?[\s:|-]+\|?\s*$/.test(line)) return "";  // separator row
    const cells = line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|");
    return "<tr>" + cells.map((c) => `<td>${inline(c.trim())}</td>`).join("") + "</tr>";
}

/* LCS-based line diff → aligned rows {left, right, type: same|added|removed}. */
function diffLines(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
    for (let i = m - 1; i >= 0; i--) {
        for (let j = n - 1; j >= 0; j--) {
            dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    const rows = [];
    let i = 0, j = 0;
    while (i < m && j < n) {
        if (a[i] === b[j]) { rows.push({ left: a[i], right: b[j], type: "same" }); i++; j++; }
        else if (dp[i + 1][j] >= dp[i][j + 1]) { rows.push({ left: a[i], right: null, type: "removed" }); i++; }
        else { rows.push({ left: null, right: b[j], type: "added" }); j++; }
    }
    while (i < m) rows.push({ left: a[i++], right: null, type: "removed" });
    while (j < n) rows.push({ left: null, right: b[j++], type: "added" });
    return rows;
}


/* ══════════════════════════════════════════════════════════════
   SYSTEM STATUS (model load + indexing)
═══════════════════════════════════════════════════════════════ */
let systemReady = false;
let statusPollTimer = null;

async function pollStatus() {
    try {
        const res  = await fetch(STATUS_URL);
        const data = await res.json();

        if (data.phase === "ready") {
            systemReady = true;
            statusBanner.hidden = true;
            setControlsDisabled(false);
            setStatus("Hazır", "");
            if (statusPollTimer) { clearInterval(statusPollTimer); statusPollTimer = null; }
            return;
        }

        // Still loading models or indexing — block input and show message.
        systemReady = false;
        statusBanner.hidden = false;
        statusBannerText.textContent = data.message || "Sistem hazırlanıyor…";
        setControlsDisabled(true);
        setStatus(data.phase === "indexing" ? "İndeksleniyor" : "Yükleniyor", "running");
    } catch {
        // Backend not reachable yet — keep trying quietly.
        statusBanner.hidden = false;
        statusBannerText.textContent = "Sunucuya bağlanılıyor…";
        setControlsDisabled(true);
    }
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

    const isWeb = results.some((r) => r.origin === "web");
    retrievalLabel.textContent = isWeb ? `🌐 ${results.length} web kaynağı` : `${results.length} doküman`;

    if (!results.length) {
        retrievalList.innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">travel_explore</span>
                <p>Bu sorgu için doküman bulunamadı.</p>
            </div>`;
        return;
    }

    results.forEach((result, index) => {
        const card     = document.createElement("article");
        const score    = typeof result.score === "number" ? result.score.toFixed(4) : "n/a";

        if (result.origin === "web") {
            card.className = "retrieval-card web-source";
            const url = result.source || "";
            let linkHtml = "";
            if (url && /^https?:\/\//i.test(url)) {
                linkHtml = `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="web-link">${escapeHtml(url)}</a>`;
            } else {
                linkHtml = `<span class="web-link">${escapeHtml(url || "invalid URL")}</span>`;
            }
            card.innerHTML = `
                <div class="retrieval-head">
                    <span class="material-symbols-outlined">public</span>
                    <strong>${escapeHtml(result.title || `Web ${index + 1}`)}</strong>
                    <span class="relevance-pill web-pill">internet</span>
                </div>
                ${linkHtml}
                <p>${escapeHtml(result.content || "")}</p>`;
        } else {
            card.className   = `retrieval-card ${result.relevant ? "relevant" : "not-relevant"}`;
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
        }
        retrievalList.appendChild(card);
    });
}

// Render the answer text. When it was sourced from the web, turn [n] citation
// tags into blue links pointing at the matching web source.
function renderAnswer(contentDiv, text, webSources, searchMeta) {
    if (webSources && webSources.length) {
        contentDiv.innerHTML = linkifyCitations(text, webSources);
        return;
    }
    // Render the answer as Markdown so structure (lists/headings/tables) and
    // inline figures (![](/images/...)) show up in the chat.
    contentDiv.innerHTML = renderMarkdown(text);
    // Document-grounded citations [1],[2] → clickable, opens a panel showing
    // the original PDF page with the cited text highlighted (fosforlu kalem).
    if (searchMeta && searchMeta.length) {
        linkifyDocCitations(contentDiv, searchMeta);
    }
}

/* For document-grounded answers, [n] citations map to the search_metadata
   entries. Clicking one opens a side panel with the original PDF page rendered
   to a PNG and the cited phrase highlighted in yellow. */
function linkifyDocCitations(contentDiv, sources) {
    const walker = document.createTreeWalker(contentDiv, NodeFilter.SHOW_TEXT, null);
    const textNodes = [];
    let node;
    while ((node = walker.nextNode())) {
        if (/\[\d+\]/.test(node.nodeValue)) textNodes.push(node);
    }
    const re = /\[(\d+)\]/g;
    textNodes.forEach((tn) => {
        const frag = document.createDocumentFragment();
        let last = 0;
        const val = tn.nodeValue;
        re.lastIndex = 0;
        let m;
        while ((m = re.exec(val))) {
            const n = parseInt(m[1], 10);
            const src = sources[n - 1];
            if (last < m.index) frag.appendChild(document.createTextNode(val.slice(last, m.index)));
            if (src) {
                const a = document.createElement("a");
                a.className = "cite-link cite-doc";
                a.textContent = `[${n}]`;
                a.title = src.title || src.source || "Atıf";
                a.href = "#";
                a.addEventListener("click", (e) => {
                    e.preventDefault();
                    openCitationPanel(src, sources);
                });
                frag.appendChild(a);
            } else {
                frag.appendChild(document.createTextNode(m[0]));
            }
            last = m.index + m[0].length;
        }
        if (last < val.length) frag.appendChild(document.createTextNode(val.slice(last)));
        tn.parentNode.replaceChild(frag, tn);
    });
}

/* ── Citation panel: original PDF page + highlighted snippet ────── */
let citePanel = null;

function ensureCitePanel() {
    if (citePanel) return citePanel;
    citePanel = document.createElement("div");
    citePanel.className = "cite-panel";
    citePanel.innerHTML = `
        <div class="cite-panel-head">
            <div class="cite-panel-title"><span class="material-symbols-outlined">menu_book</span> <span id="citePanelTitleText">Atıf</span></div>
            <div class="cite-panel-nav">
                <button id="citePrev" type="button" title="Önceki sayfa"><span class="material-symbols-outlined">chevron_left</span></button>
                <span id="citePageLabel">—</span>
                <button id="citeNext" type="button" title="Sonraki sayfa"><span class="material-symbols-outlined">chevron_right</span></button>
            </div>
            <button id="citePanelClose" type="button" class="ghost-button" title="Kapat"><span class="material-symbols-outlined">close</span></button>
        </div>
        <div class="cite-panel-loading">Yükleniyor…</div>
        <div class="cite-panel-body"></div>`;
    document.body.appendChild(citePanel);
    citePanel.querySelector("#citePanelClose").addEventListener("click", closeCitePanel);
    citePanel.querySelector("#citePrev").addEventListener("click", () => citeNav(-1));
    citePanel.querySelector("#citeNext").addEventListener("click", () => citeNav(1));
    citePanel.addEventListener("click", (e) => { if (e.target === citePanel) closeCitePanel(); });
    return citePanel;
}

let citeState = null; // {source, totalPages, currentPage}

// Open the viewer for a clicked citation. Renders the WHOLE original PDF with
// every retrieved chunk of that source highlighted, scrolls to the clicked
// page, and docks the panel so the workspace reflows beside it.
async function openCitationPanel(src, sources = []) {
    const panel = ensureCitePanel();
    panel.classList.add("is-open");
    document.body.classList.add("cite-open");
    const body = panel.querySelector(".cite-panel-body");
    const loading = panel.querySelector(".cite-panel-loading");
    body.innerHTML = "";
    body.onscroll = null;
    loading.style.display = "block";
    panel.querySelector("#citePanelTitleText").textContent = src.source || src.title || "Atıf";
    panel.querySelector("#citePageLabel").textContent = "…";

    const source = src.source || "";
    const focusSnippet = src.content || src.snippet || "";
    // Every retrieved chunk that belongs to this same source document.
    const snippets = (sources || [])
        .filter((s) => (s.source || "") === source)
        .map((s) => s.content || s.snippet || "")
        .filter(Boolean);
    if (!snippets.length && focusSnippet) snippets.push(focusSnippet);

    citeState = { source, totalPages: null, currentPage: 0 };

    const data = await fetchCiteDoc(source, snippets, focusSnippet);
    loading.style.display = "none";
    if (data.error || !Array.isArray(data.pages) || data.pages.length === 0) {
        const msg = data.error || "Atıf görüntüsü yüklenemedi.";
        body.innerHTML = `<div class="cite-panel-error"><span class="material-symbols-outlined">error</span> ${escapeHtml(msg)}</div>`;
        citeState = null;
        return;
    }
    citeState.totalPages = data.total_pages;
    renderCiteDoc(data);
}

async function fetchCiteDoc(source, snippets, focusSnippet) {
    try {
        const res = await fetch("/cite/doc", {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ source, snippets, focus_snippet: focusSnippet }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            return { error: data.error || "Atıf görüntüsü oluşturulamadı." };
        }
        return data;
    } catch (e) {
        return { error: "Sunucuya ulaşılamadı." };
    }
}

function renderCiteDoc(data) {
    const panel = ensureCitePanel();
    const body = panel.querySelector(".cite-panel-body");
    body.innerHTML = "";

    (data.pages || []).forEach((p) => {
        const fig = document.createElement("div");
        fig.className = "cite-page";
        fig.dataset.page = p.page;
        const img = document.createElement("img");
        img.className = "cite-page-img";
        img.src = p.image_url;
        img.alt = `Sayfa ${p.page}`;
        img.loading = "lazy";
        fig.appendChild(img);
        body.appendChild(fig);
    });

    const focusIdx = Math.max(0, Math.min(data.focus_page_index ?? 0, (data.total_pages ?? 1) - 1));
    // Wait for the focus page image to lay out, then jump to it.
    const focusEl = body.querySelectorAll(".cite-page")[focusIdx];
    const jump = () => scrollCiteToPage(focusIdx);
    const focusImg = focusEl && focusEl.querySelector("img");
    if (focusImg && !focusImg.complete) {
        focusImg.addEventListener("load", jump, { once: true });
        focusImg.addEventListener("error", jump, { once: true });
    } else {
        requestAnimationFrame(jump);
    }
    setupCiteScrollSpy();
    updateCitePageLabel(focusIdx);
}

function setupCiteScrollSpy() {
    const body = citePanel.querySelector(".cite-panel-body");
    body.onscroll = () => {
        const pages = [...body.querySelectorAll(".cite-page")];
        const threshold = body.scrollTop + body.clientHeight * 0.35;
        let cur = 0;
        pages.forEach((p, i) => { if (p.offsetTop <= threshold) cur = i; });
        updateCitePageLabel(cur);
    };
}

function scrollCiteToPage(index) {
    const body = citePanel.querySelector(".cite-panel-body");
    const el = body.querySelectorAll(".cite-page")[index];
    if (el) body.scrollTo({ top: Math.max(0, el.offsetTop - 16), behavior: "auto" });
    updateCitePageLabel(index);
}

function updateCitePageLabel(index) {
    if (!citeState) return;
    citeState.currentPage = index;
    citePanel.querySelector("#citePageLabel").textContent =
        citeState.totalPages ? `${index + 1} / ${citeState.totalPages}` : `${index + 1}`;
}

function citeNav(delta) {
    if (!citeState) return;
    const next = Math.max(0, Math.min((citeState.currentPage ?? 0) + delta, (citeState.totalPages ?? 1) - 1));
    scrollCiteToPage(next);
}

function closeCitePanel() {
    if (citePanel) citePanel.classList.remove("is-open");
    document.body.classList.remove("cite-open");
}

function linkifyCitations(text, sources) {
    return escapeHtml(text).replace(/\[(\d+)\]/g, (match, n) => {
        const src = sources[parseInt(n, 10) - 1];
        const url = src && src.source;
        if (url && /^https?:\/\//i.test(url)) {
            return `<a class="cite-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(src.title || url)}">[${n}]</a>`;
        }
        return match;
    });
}

// Weak RAG match: ask the user before falling back to a web search.
function showWebSearchPrompt(botMessageDiv, query) {
    const contentDiv = botMessageDiv.querySelector(".message-content");
    contentDiv.textContent =
        "Bu soruyla ilgili belgelerde yeterli bilgi bulamadım. İnternette aramamı ister misiniz?";

    const actions = document.createElement("div");
    actions.className = "web-approval";

    const yesBtn = document.createElement("button");
    yesBtn.className = "web-approval-btn yes";
    yesBtn.innerHTML = `<span class="material-symbols-outlined">travel_explore</span> Evet, internette ara`;

    const noBtn = document.createElement("button");
    noBtn.className = "web-approval-btn no";
    noBtn.textContent = "Hayır, gerek yok";

    yesBtn.addEventListener("click", () => {
        actions.remove();
        logEvent("retrieval", "Kullanıcı internet aramasını onayladı.");
        runQuery(query, { allowWeb: true, echoUser: false });
    });
    noBtn.addEventListener("click", () => {
        actions.remove();
        contentDiv.textContent = "Anlaşıldı, internette arama yapılmadı.";
        logEvent("retrieval", "Kullanıcı internet aramasını reddetti.");
    });

    actions.appendChild(yesBtn);
    actions.appendChild(noBtn);
    botMessageDiv.appendChild(actions);
    logEvent("retrieval", "Belgelerde bulunamadı — kullanıcı onayı bekleniyor.");
    scrollToBottom();
}

/* ══════════════════════════════════════════════════════════════
   CHAT MESSAGES
═══════════════════════════════════════════════════════════════ */
function addMessage(text, type, images = []) {
    const div     = document.createElement("div");
    div.className = `message ${type}-message`;

    const avatar       = document.createElement("div");
    avatar.className   = "message-avatar";
    avatar.textContent = type === "user" ? "YOU" : "AI";

    const content       = document.createElement("div");
    content.className   = "message-content";
    content.textContent = text;

    if (images && images.length) {
        const gallery = document.createElement("div");
        gallery.className = "message-images";
        images.forEach((src) => {
            const img = document.createElement("img");
            img.src = src;
            img.loading = "lazy";
            gallery.appendChild(img);
        });
        content.appendChild(gallery);
    }

    div.appendChild(avatar);
    div.appendChild(content);
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

/* Render figures retrieved from the corpus, anchored under the answer. */
function showContextImages(botMessageDiv, images) {
    if (!images || !images.length) return;
    const gallery = document.createElement("div");
    gallery.className = "context-images";
    images.forEach((item) => {
        const link = document.createElement("a");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.title = item.source || "";
        const img = document.createElement("img");
        img.src = item.url;
        img.loading = "lazy";
        img.alt = item.name || "figure";
        link.appendChild(img);
        gallery.appendChild(link);
    });
    botMessageDiv.appendChild(gallery);
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
    logEvent("idle", "Sistem durumu kontrol ediliyor…");

    // Gate input until models are loaded and the index is built.
    pollStatus();
    statusPollTimer = setInterval(pollStatus, 3000);

    loadLibrary();
});
