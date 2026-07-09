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
const tabEditor         = document.getElementById("tabEditor");
const libraryList       = document.getElementById("libraryList");
const libraryCountTag   = document.getElementById("libraryCountTag");
const libraryViewTabs   = document.getElementById("libraryViewTabs");
const libraryDocTitle   = document.getElementById("libraryDocTitle");
const libraryDocBody    = document.getElementById("libraryDocBody");
const libraryDownloads  = document.getElementById("libraryDownloads");
const editorView        = document.getElementById("editorView");
const editorFileList    = document.getElementById("editorFileList");
const editorCountTag    = document.getElementById("editorCountTag");
const editorOutlinePanel = document.getElementById("editorOutlinePanel");
const editorOutlineToggleBtn = document.getElementById("editorOutlineToggleBtn");
const insertTOCBtn      = document.getElementById("insertTOCBtn");
const outlineList       = document.getElementById("outlineList");
const editorPage        = document.getElementById("editorPage");
const editorToolbar     = document.getElementById("editorToolbar");
const editorSaveStatus  = document.getElementById("editorSaveStatus");
const editorChatForm    = document.getElementById("editorChatForm");
const editorUserInput   = document.getElementById("editorUserInput");
const editorSendButton  = document.getElementById("editorSendButton");
const editorDownloadDocx = document.getElementById("editorDownloadDocx");
const editorDownloadPdf = document.getElementById("editorDownloadPdf");
const editorDownloadMd  = document.getElementById("editorDownloadMd");
const editorDownloadHtml = document.getElementById("editorDownloadHtml");
const editorHistoryBtn  = document.getElementById("editorHistoryBtn");
const historyModal      = document.getElementById("historyModal");
const historyClose      = document.getElementById("historyClose");
const historyList       = document.getElementById("historyList");
const historySub        = document.getElementById("historySub");
const reportChatMessages = document.getElementById("reportChatMessages");
const reportTools       = document.getElementById("reportTools");
const reportToolsToggleBtn = document.getElementById("reportToolsToggleBtn");
const newReportBtn      = document.getElementById("newReportBtn");
const newReportModal    = document.getElementById("newReportModal");
const newReportClose    = document.getElementById("newReportClose");
const newReportTitle    = document.getElementById("newReportTitle");
const newReportCreate   = document.getElementById("newReportCreate");
const newReportStatus   = document.getElementById("newReportStatus");
const templateGrid      = document.getElementById("templateGrid");
const newReportUseAI    = document.getElementById("newReportUseAI");
const aiGenPromptWrap   = document.getElementById("aiGenPromptWrap");
const newReportAIPrompt = document.getElementById("newReportAIPrompt");
const aiGenStatusWrap   = document.getElementById("aiGenStatusWrap");
const aiGenStatusText   = document.getElementById("aiGenStatusText");
const aiGenProgressFill = document.getElementById("aiGenProgressFill");
const aiGenLogs         = document.getElementById("aiGenLogs");
const newReportActions  = document.getElementById("newReportActions");
const blockGrid         = document.getElementById("blockGrid");
const evidenceList      = document.getElementById("evidenceList");
const evidenceCount     = document.getElementById("evidenceCount");
const libraryAssets     = document.getElementById("libraryAssets");
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
    supervisor: "Making routing decision",
    researcher: "Searching vector database",
    writer:     "Writing evidence-based answer",
    reviewer:   "Quality-checking the answer",
    editor:     "Updating knowledge and re-indexing",
};

/* ── Auth state ──────────────────────────────────────────────── */
const AUTH_LOGIN_URL = "/auth/login";
const AUTH_SIGNUP_URL = "/auth/signup";
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
        remove.setAttribute("aria-label", "Remove");
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
    updateProgress(8, "Request received", "Input staging");
    logEvent("request", "User request received. Running graph.");
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
            let detail = "You must be signed in to use @update.";
            try { const p = await response.json(); if (p?.error) detail = p.error; } catch {}
            throw new Error(detail, { cause: "auth" });
        }

        if (!response.ok || !response.body) {
            let detail = "";
            try { const p = await response.json(); detail = p?.error ? ` ${p.error}` : ""; } catch {}
            throw new Error(`API request failed.${detail}`);
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
                    logEvent("retrieval", `Fetched ${webResults.length} sources from the web.`);
                } else {
                    logEvent("retrieval", `${results.length} document scores retrieved.`);
                }
            }

            /* figures anchored to the retrieved context */
            if (event.event === "context_images") {
                const images = safeJson(event.data) || [];
                showContextImages(botMessageDiv, images);
                if (images.length) logEvent("retrieval", `${images.length} related images found.`);
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
                    
                    // If editor view is active and matches current source, try inline diff
                    if (!editorView.hidden && editorDoc && (payload.file === editorDoc.source || payload.source === editorDoc.source)) {
                        const inlineOk = showInlineEditPreview(payload);
                        if (inlineOk) {
                            logEvent("editor", `Change preview ready in editor (${payload.file || ""}).`);
                            continue;
                        }
                    }
                    
                    showEditPreview(botMessageDiv, payload);
                }
            }

            /* streaming tokens from writer */
            if (event.event === "token") {
                streamingStarted = true;
                fullText += event.data;
                contentDiv.innerHTML = renderMarkdown(fullText);
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
                throw new Error(payload?.error || "Unknown stream error.");
            }
        }

        if (!fullText.trim() && !awaitingApproval && !awaitingEditApproval) {
            contentDiv.textContent = "Stream completed but no response text was generated.";
        }
        if (!completed) markComplete();

    } catch (error) {
        typingIndicator.remove();
        contentDiv.classList.remove("is-streaming");
        if (!fullText.trim()) {
            contentDiv.textContent = `Sorry, an error occurred. ${error.message || "Please try again."}`;
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
        card.title = "View updated document";
        card.addEventListener("click", (e) => {
            if (e.target.closest("a")) return;  // let the download link work normally
            openUpdatedDoc(payload.file);
        });
    }

    const title = document.createElement("div");
    title.className = "edit-result-title";
    title.innerHTML = `<span class="material-symbols-outlined">drive_file_rename_outline</span> ${payload.file || "Workspace updated"}`;
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
        link.innerHTML = `<span class="material-symbols-outlined">download</span> Download updated PDF`;
        card.appendChild(link);
    }

    if (payload.file) {
        const hint = document.createElement("div");
        hint.className = "edit-result-hint";
        hint.innerHTML = `<span class="material-symbols-outlined">open_in_new</span> Click to see the change`;
        card.appendChild(hint);
    }

    botMessageDiv.appendChild(card);
    logEvent("editor", `Workspace updated: ${payload.file || ""}`);
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
        <span class="edit-preview-title">Change preview</span>
        <span class="edit-preview-file">${escapeHtml(payload.file || "")}</span>`;
    card.appendChild(header);

    if (payload.instruction) {
        const instr = document.createElement("p");
        instr.className = "edit-preview-instruction";
        instr.innerHTML = `<strong>Instruction:</strong> ${escapeHtml(payload.instruction)}`;
        card.appendChild(instr);
    }

    const diff = payload.diff || [];
    if (!diff.length) {
        const note = document.createElement("p");
        note.className = "edit-preview-empty";
        note.textContent = "No change detected.";
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
    meta.textContent = `+${added} lines · −${removed} lines`;
    card.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "edit-preview-actions";

    const approveBtn = document.createElement("button");
    approveBtn.type = "button";
    approveBtn.className = "primary-button edit-approve";
    approveBtn.innerHTML = `<span class="material-symbols-outlined">check</span> Approve`;
    approveBtn.disabled = false;

    const rejectBtn = document.createElement("button");
    rejectBtn.type = "button";
    rejectBtn.className = "ghost-button edit-reject";
    rejectBtn.innerHTML = `<span class="material-symbols-outlined">close</span> Reject`;

    actions.appendChild(rejectBtn);
    actions.appendChild(approveBtn);
    card.appendChild(actions);

    const status = document.createElement("div");
    status.className = "edit-preview-status";
    card.appendChild(status);

    botMessageDiv.appendChild(card);
    logEvent("editor", `Change preview ready — awaiting approval (${payload.file || ""}).`);
    scrollToBottom();

    async function approve() {
        approveBtn.disabled = true; rejectBtn.disabled = true;
        status.textContent = "Applying…"; status.className = "edit-preview-status is-pending";
        try {
            const res = await fetch("/update/apply", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ token: payload.token }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Approval failed.");
            status.innerHTML = `<span class="material-symbols-outlined">check_circle</span> ${escapeHtml(data.reply || "Change applied.")}`;
            status.className = "edit-preview-status is-ok";
            card.querySelector(".diff-viewer").classList.add("diff-applied");
            card.querySelector(".edit-preview-actions").remove();
            addMessage(data.reply, "assistant");
            loadLibrary();
        } catch (err) {
            status.textContent = `Error: ${err.message}`;
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
        status.innerHTML = `<span class="material-symbols-outlined">cancel</span> Change rejected — not applied to document.`;
        status.className = "edit-preview-status is-rejected";
        card.querySelector(".edit-preview-actions").remove();
        logEvent("editor", "Change rejected by user.");
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
    label.textContent = "Was this response helpful?";

    const upBtn   = makeFeedbackBtn("thumb_up",   "Good response",  1);
    const downBtn = makeFeedbackBtn("thumb_down", "Poor response", -1);

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
        label.textContent = "Thanks! 👍";
        sendFeedback(1, "");
    });

    downBtn.addEventListener("click", () => {
        if (submitted) return;
        submitted = true;
        lockButtons(-1);
        label.textContent = "Thanks for your feedback.";
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
    textarea.placeholder = "What could we improve? (optional)";
    textarea.rows = 2;

    const submit = document.createElement("button");
    submit.type = "button";
    submit.className = "feedback-comment-submit";
    submit.textContent = "Send";

    let sent = false;
    function finish() {
        if (sent) return;
        sent = true;
        onSubmit(textarea.value.trim());
        box.innerHTML = `<span class="feedback-comment-done">Your comment was saved.</span>`;
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
const authModalTitle  = document.getElementById("authModalTitle");
const authHint        = document.getElementById("authHint");
const authSubmit      = document.getElementById("authSubmit");
const authSwitch      = document.getElementById("authSwitch");

let authMode = "login"; // "login" | "signup"

function renderAuthState() {
    if (!authButtonLabel) return;
    authButtonLabel.textContent = isAuthenticated() ? (authUser || "Sign out") : "Sign in";
    if (authButton) authButton.title = isAuthenticated()
        ? `Signed in as ${authUser} — click to sign out`
        : "Sign in";
}

function applyAuthMode() {
    if (!authModal) return;
    if (authMode === "signup") {
        authModalTitle.textContent = "Sign Up";
        authHint.textContent = "Create the first account for this instance — it becomes admin.";
        authSubmit.textContent = "Sign up";
        authSwitch.innerHTML = 'Already have an account? <button type="button" id="authSwitchBtn" class="auth-switch-btn">Log in</button>';
    } else {
        authModalTitle.textContent = "Log In";
        authHint.textContent = "Login is required to update documents with @update.";
        authSubmit.textContent = "Log in";
        authSwitch.innerHTML = 'No account yet? <button type="button" id="authSwitchBtn" class="auth-switch-btn">Sign up</button>';
    }
    // innerHTML replace drops the old button + its listener; re-bind the new one.
    document.getElementById("authSwitchBtn").addEventListener("click", () => {
        authMode = authMode === "login" ? "signup" : "login";
        authError.hidden = true;
        applyAuthMode();
    });
}

function openAuthModal() {
    if (!authModal) return;
    authMode = "login";
    authError.hidden = true;
    authForm.reset();
    applyAuthMode();
    authModal.hidden = false;
    authUsername.focus();
}

function closeAuthModal() { if (authModal) authModal.hidden = true; }

if (authButton) {
    authButton.addEventListener("click", () => {
        if (isAuthenticated()) {
            // Toggle to logout when already signed in.
            setAuth("", "");
            logEvent("auth", "Signed out.");
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

        const url = authMode === "signup" ? AUTH_SIGNUP_URL : AUTH_LOGIN_URL;
        try {
            const res = await fetch(url, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ username, password }),
            });
            const data = await res.json();
            if (!res.ok) {
                authError.textContent = data.error || (authMode === "signup" ? "Sign-up failed." : "Sign-in failed.");
                authError.hidden = false;
                return;
            }
            setAuth(data.token, data.username || username);
            closeAuthModal();
            logEvent("auth", authMode === "signup"
                ? `Signed up as ${data.username || username} (admin).`
                : `Signed in as ${data.username || username}.`);
        } catch (err) {
            authError.textContent = "Could not reach the server.";
            authError.hidden = false;
        }
    });
}

renderAuthState();

tabChat.addEventListener("click", () => switchTab("chat"));
tabLibrary.addEventListener("click", () => switchTab("library"));
tabEditor.addEventListener("click", () => switchTab("editor"));
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
        showUploadStatus("Only .pdf and .docx files are supported.", "error");
        return;
    }

    showUploadStatus("Uploading and indexing…", "loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(UPLOAD_URL, { method: "POST", headers: authHeaders(), body: formData });
        const data = await res.json();

        if (!res.ok) {
            showUploadStatus(`Error: ${data.error || res.statusText}`, "error");
            return;
        }

        showUploadStatus(
            `✓ "${escapeHtml(data.filename)}" uploaded successfully — ${data.chunks_added} chunks indexed.`,
            "success",
        );
        logEvent("upload", `${data.filename} → ${data.chunks_added} chunks added.`);
        loadDocumentList();
        loadLibrary();
        loadEditorFileList();
    } catch (err) {
        showUploadStatus(`Upload failed: ${err.message}`, "error");
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
    docCountTag.textContent = "Loading…";
    docList.innerHTML = "";

    try {
        const res  = await fetch(ADMIN_DOCS_URL, { headers: authHeaders() });
        const data = await res.json();
        const docs = data.documents || [];

        docCountTag.textContent = `${docs.length} sources`;

        if (!docs.length) {
            docList.innerHTML = `
                <div class="doc-list-empty">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>No documents yet.</p>
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
            delBtn.title = "Delete";
            delBtn.innerHTML = `<span class="material-symbols-outlined">delete</span>`;
            delBtn.addEventListener("click", () => deleteDocument(doc.source, row));

            row.appendChild(info);
            row.appendChild(delBtn);
            docList.appendChild(row);
        });
    } catch {
        docCountTag.textContent = "Error";
        docList.innerHTML = `<div class="doc-list-empty"><p>Could not load documents.</p></div>`;
    }
}

async function deleteDocument(source, rowEl) {
    if (!confirm(`Are you sure you want to delete "${source.split("/").pop()}"?`)) return;

    rowEl.classList.add("is-deleting");
    try {
        const res  = await fetch(`${ADMIN_DOCS_URL}/${encodeURIComponent(source)}`, { method: "DELETE", headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) {
            alert(data.error || "Delete failed.");
            rowEl.classList.remove("is-deleting");
            return;
        }
        rowEl.remove();
        logEvent("upload", `${source.split("/").pop()} deleted (${data.chunks_deleted} chunks).`);
        const remaining = docList.querySelectorAll(".doc-row").length;
        docCountTag.textContent = `${remaining} sources`;
        if (!remaining) {
            docList.innerHTML = `
                <div class="doc-list-empty">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>No documents yet.</p>
                </div>`;
        }
        loadLibrary();
    } catch {
        alert("Delete request failed.");
        rowEl.classList.remove("is-deleting");
    }
}

/* ══════════════════════════════════════════════════════════════
   LIBRARY TAB (view + compare originals vs workspace)
═══════════════════════════════════════════════════════════════ */
let libraryDoc = null;   // { source, original, workspace, original_url, workspace_pdf_url }
let libraryView_ = "workspace";
let editorDoc = null;    // { source, markdown }

function switchTab(tab) {
    chatView.hidden = tab !== "chat";
    libraryView.hidden = tab !== "library";
    editorView.hidden = tab !== "editor";
    closeCitePanel();

    tabChat.classList.toggle("is-active", tab === "chat");
    tabLibrary.classList.toggle("is-active", tab === "library");
    tabEditor.classList.toggle("is-active", tab === "editor");
    
    if (tab === "library") {
        loadLibrary();
    } else if (tab === "editor") {
        loadEditorFileList();
        loadLibraryAssets();
    }
}

async function fetchWorkspaceDocs() {
    const res  = await fetch(`${ADMIN_DOCS_URL}?channel=workspace`, { headers: authHeaders() });
    if (!res.ok) throw new Error("fetch failed");
    const data = await res.json();
    return data.documents || [];
}

function renderDocRows(container, docs, activeSource, onSelect, opts = {}) {
    container.innerHTML = "";
    appendDocRows(container, docs, activeSource, onSelect, opts);
}

function appendDocRows(container, docs, activeSource, onSelect, opts = {}) {
    docs.forEach((doc) => {
        const name = doc.title || doc.source;
        const icon = doc.kind === "report" ? (doc.generated ? "auto_awesome" : "description") : "upload_file";
        const row = document.createElement("div");
        row.className = "library-row";
        row.dataset.source = doc.source;
        if (activeSource === doc.source) row.classList.add("is-active");
        row.innerHTML = `
            <span class="material-symbols-outlined">${icon}</span>
            <span class="library-row-text">
                <span class="library-row-name">${escapeHtml(name)}</span>
                <span class="library-row-meta">${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}</span>
            </span>`;
        row.addEventListener("click", (e) => {
            if (e.target.closest(".row-menu-btn") || e.target.closest(".row-menu")) return;
            onSelect(doc.source);
        });
        if (opts.withMenu) attachRowMenu(row, doc);
        container.appendChild(row);
    });
}

function docSectionHeader(label, count) {
    const h = document.createElement("div");
    h.className = "doc-group-head";
    h.innerHTML = `${escapeHtml(label)} <span class="doc-group-count">${count}</span>`;
    return h;
}

// A per-row "⋯" menu: Rename / Duplicate (reports only) + Delete (all).
function attachRowMenu(row, doc) {
    const isReport = doc.kind === "report";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "row-menu-btn";
    btn.title = "Actions";
    btn.innerHTML = `<span class="material-symbols-outlined">more_vert</span>`;
    row.appendChild(btn);

    btn.addEventListener("click", (e) => {
        e.stopPropagation();
        closeAllRowMenus();
        const menu = document.createElement("div");
        menu.className = "row-menu";
        const items = [];
        if (isReport) {
            items.push(`<button type="button" data-act="rename"><span class="material-symbols-outlined">edit</span> Rename</button>`);
            items.push(`<button type="button" data-act="duplicate"><span class="material-symbols-outlined">content_copy</span> Duplicate</button>`);
        }
        items.push(`<button type="button" data-act="delete" class="danger"><span class="material-symbols-outlined">delete</span> Delete</button>`);
        menu.innerHTML = items.join("");
        row.appendChild(menu);
        menu.addEventListener("click", (ev) => {
            const act = ev.target.closest("button")?.dataset.act;
            if (act) { ev.stopPropagation(); closeAllRowMenus(); handleRowAction(act, doc); }
        });
        document.addEventListener("click", closeAllRowMenus, { once: true });
    });
}

function closeAllRowMenus() {
    document.querySelectorAll(".row-menu").forEach((m) => m.remove());
}

async function handleRowAction(act, doc) {
    const src = doc.source;
    try {
        if (act === "rename") {
            const title = window.prompt("New report title:", doc.title || doc.source.replace(/\.md$/, ""));
            if (!title || !title.trim()) return;
            const res = await fetch(`/documents/${encodeURIComponent(src)}/rename`, {
                method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ title: title.trim() })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Rename failed.");
            await loadEditorFileList();
            if (editorDoc && editorDoc.source === src) await selectEditorDoc(data.source);
        } else if (act === "duplicate") {
            const res = await fetch(`/documents/${encodeURIComponent(src)}/duplicate`, {
                method: "POST", headers: authHeaders({ "Content-Type": "application/json" })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Duplicate failed.");
            await loadEditorFileList();
            await selectEditorDoc(data.source);
        } else if (act === "delete") {
            if (!window.confirm(`Delete "${doc.title || doc.source}"? This cannot be undone.`)) return;
            const res = await fetch(`/admin/documents/${encodeURIComponent(src)}`, {
                method: "DELETE", headers: authHeaders()
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Delete failed.");
            if (editorDoc && editorDoc.source === src) {
                editorDoc = null;
                editorPage.hidden = true;
                editorDownloadDocx.disabled = true;
                if (editorDownloadPdf) editorDownloadPdf.disabled = true;
            }
            await loadEditorFileList();
            loadLibrary();
        }
    } catch (err) {
        alert(err.message);
    }
}

async function loadLibrary() {
    try {
        const docs = await fetchWorkspaceDocs();
        libraryCountTag.textContent = String(docs.length);
        if (!docs.length) {
            libraryList.innerHTML = `
                <div class="empty-state">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>No documents yet. Upload a PDF/DOCX to get started.</p>
                </div>`;
            return;
        }
        renderDocRows(libraryList, docs, libraryDoc ? libraryDoc.source : null, selectDoc);
    } catch {
        libraryCountTag.textContent = "!";
        libraryList.innerHTML = `<div class="empty-state"><p>Could not load documents.</p></div>`;
    }
}

async function loadEditorFileList() {
    try {
        const docs = await fetchWorkspaceDocs();
        editorCountTag.textContent = String(docs.length);
        if (!docs.length) {
            editorFileList.innerHTML = `
                <div class="empty-state">
                    <span class="material-symbols-outlined">folder_open</span>
                    <p>No reports yet. Create one with "New" or upload a PDF/DOCX.</p>
                </div>`;
            return;
        }
        const active = editorDoc ? editorDoc.source : null;
        const reports = docs.filter((d) => d.kind === "report");
        const uploads = docs.filter((d) => d.kind !== "report");
        editorFileList.innerHTML = "";
        if (reports.length) {
            editorFileList.appendChild(docSectionHeader("Reports", reports.length));
            appendDocRows(editorFileList, reports, active, selectEditorDoc, { withMenu: true });
        }
        if (uploads.length) {
            editorFileList.appendChild(docSectionHeader("Uploaded documents", uploads.length));
            appendDocRows(editorFileList, uploads, active, selectEditorDoc, { withMenu: true });
        }
    } catch {
        editorCountTag.textContent = "!";
        editorFileList.innerHTML = `<div class="empty-state"><p>Could not load documents.</p></div>`;
    }
}


async function selectDoc(source) {
    libraryDocTitle.textContent = source;
    libraryViewTabs.hidden = false;
    libraryDownloads.innerHTML = "";
    libraryDocBody.innerHTML = `<div class="compare-loading">Loading…</div>`;
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
            workspace_docx_url: ws.workspace_docx_url || "",
        };

        let dl = "";
        if (libraryDoc.original_url)
            dl += `<a class="ghost-button" href="${libraryDoc.original_url}" target="_blank" rel="noopener"><span class="material-symbols-outlined">picture_as_pdf</span> Original</a>`;
        if (libraryDoc.workspace_pdf_url)
            dl += `<a class="ghost-button" href="${libraryDoc.workspace_pdf_url}" target="_blank" rel="noopener"><span class="material-symbols-outlined">download</span> Updated PDF</a>`;
        if (libraryDoc.workspace_docx_url)
            dl += `<a class="ghost-button" href="${libraryDoc.workspace_docx_url}" target="_blank" rel="noopener"><span class="material-symbols-outlined">description</span> Updated Word</a>`;
        libraryDownloads.innerHTML = dl;

        renderDocView(libraryView_);
    } catch {
        libraryDocBody.innerHTML = `<div class="compare-loading">Could not load document.</div>`;
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
            ${identical ? `<div class="compare-note">Original and workspace are the same (no edits yet).</div>` : ""}
            <div class="compare-grid">
                <div class="compare-col">
                    <div class="compare-col-head">Original <small>(read-only)</small></div>
                    <div class="compare-pane">${left}</div>
                </div>
                <div class="compare-col">
                    <div class="compare-col-head">Workspace <small>(edited)</small></div>
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
        .replace(/~~([^~]+)~~/g, "<del>$1</del>")
        .replace(/&lt;u&gt;([\s\S]*?)&lt;\/u&gt;/g, "<u>$1</u>")
        .replace(/&lt;mark&gt;([\s\S]*?)&lt;\/mark&gt;/g, "<mark>$1</mark>")
        .replace(/&lt;mark\s+style=&quot;background:\s*([^&;]+);?&quot;&gt;([\s\S]*?)&lt;\/mark&gt;/g, '<mark style="background: $1;">$2</mark>')
        .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
        .replace(/\[([^\]]+)\]\(cite:\/\/([^?)]+)\?snippet=([^)]+)\)/g, (match, text, filename, snippet) => {
            const decodedSnippet = decodeURIComponent(snippet.replace(/\+/g, " "));
            return `<a href="#" class="citation-link" data-filename="${escapeHtml(filename)}" data-snippet="${escapeHtml(decodedSnippet)}">${escapeHtml(text)}</a>`;
        });

    for (const raw of lines) {
        const line = raw.replace(/\s+$/, "");
        if (line.trim().startsWith("```")) { inCode = !inCode; html += inCode ? "<pre><code>" : "</code></pre>"; continue; }
        if (inCode) { html += escapeHtml(raw) + "\n"; continue; }

        // Raw HTML block round-trip
        if (line.trim().startsWith("<div") || line.trim().startsWith("<img")) {
            closeList();
            html += line + "\n";
            continue;
        }

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
            setStatus("Ready", "");
            if (statusPollTimer) { clearInterval(statusPollTimer); statusPollTimer = null; }
            return;
        }

        // Still loading models or indexing — block input and show message.
        systemReady = false;
        statusBanner.hidden = false;
        statusBannerText.textContent = data.message || "System loading…";
        setControlsDisabled(true);
        setStatus(data.phase === "indexing" ? "Indexing" : "Loading", "running");
    } catch {
        // Backend not reachable yet — keep trying quietly.
        statusBanner.hidden = false;
        statusBannerText.textContent = "Connecting to server…";
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
    updateProgress(100, "Answer ready", "Complete");
    currentAgentLabel.textContent = "Complete";
    setStatus("Complete", "complete");
    logEvent("done", "Stream completed, answer delivered.");
}

function markError(message) {
    agentCards.forEach((c) => c.classList.remove("is-active"));
    const lastAgent = Array.from(visitedAgents).pop();
    const errCard = lastAgent ? document.querySelector(`[data-agent="${lastAgent}"]`) : null;
    if (errCard) errCard.classList.add("has-error");
    updateProgress(100, "An error occurred", "Error");
    currentAgentLabel.textContent = "Error";
    setStatus("Error", "error");
    logEvent("error", message || "Stream ended with an error.");
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
            <p>Evidence will appear here when a new query runs.</p>
        </div>`;
    retrievalLabel.textContent = "Empty";
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
    retrievalLabel.textContent = isWeb ? `🌐 ${results.length} web sources` : `${results.length} documents`;

    if (!results.length) {
        retrievalList.innerHTML = `
            <div class="empty-state">
                <span class="material-symbols-outlined">travel_explore</span>
                <p>No documents found for this query.</p>
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

/* Build a deduplicated citation source list that matches the backend's
   format_docs_with_parents numbering (which deduplicates by parent_id).
   [1] in the LLM answer → citationSources[0], etc. */
function buildCitationSources(sources) {
    const seen = new Set();
    return sources.filter((src) => {
        const pid = src.parent_id;
        if (pid) {
            if (seen.has(pid)) return false;
            seen.add(pid);
        }
        return true;
    });
}

/* For document-grounded answers, [n] citations map to the search_metadata
   entries. Clicking one opens a side panel with the original PDF page rendered
   to a PNG and the cited phrase highlighted in yellow. */
function linkifyDocCitations(contentDiv, sources) {
    const citationSources = buildCitationSources(sources);
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
            const src = citationSources[n - 1];
            if (last < m.index) frag.appendChild(document.createTextNode(val.slice(last, m.index)));
            if (src) {
                const a = document.createElement("a");
                a.className = "cite-link cite-doc";
                a.textContent = `[${n}]`;
                a.title = src.title || src.source || "Citation";
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
            <div class="cite-panel-title"><span class="material-symbols-outlined">menu_book</span> <span id="citePanelTitleText">Citation</span></div>
            <div class="cite-panel-modes" role="tablist">
                <button id="citeModePdf" type="button" class="cite-mode-btn is-active" data-mode="pdf"><span class="material-symbols-outlined">picture_as_pdf</span> PDF</button>
                <button id="citeModeDiff" type="button" class="cite-mode-btn" data-mode="diff"><span class="material-symbols-outlined">difference</span> Changes</button>
            </div>
            <div class="cite-panel-nav">
                <button id="citePrev" type="button" title="Previous page"><span class="material-symbols-outlined">chevron_left</span></button>
                <span id="citePageLabel">—</span>
                <button id="citeNext" type="button" title="Next page"><span class="material-symbols-outlined">chevron_right</span></button>
            </div>
            <button id="citePanelClose" type="button" class="ghost-button" title="Close"><span class="material-symbols-outlined">close</span></button>
        </div>
        <div class="cite-panel-loading">Loading…</div>
        <div class="cite-panel-body"></div>`;
    document.body.appendChild(citePanel);
    citePanel.querySelector("#citePanelClose").addEventListener("click", closeCitePanel);
    citePanel.querySelector("#citePrev").addEventListener("click", () => citeNav(-1));
    citePanel.querySelector("#citeNext").addEventListener("click", () => citeNav(1));
    citePanel.querySelector("#citeModePdf").addEventListener("click", () => setCiteView("pdf"));
    citePanel.querySelector("#citeModeDiff").addEventListener("click", () => setCiteView("diff"));
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
    panel.querySelector("#citePanelTitleText").textContent = src.source || src.title || "Citation";
    panel.querySelector("#citePageLabel").textContent = "…";

    const source = src.source || "";
    const focusSnippet = src.parent_content || src.content || src.snippet || "";
    // Every retrieved chunk that belongs to this same source document.
    // Use parent_content (full section) when available so the entire retrieved
    // section is highlighted in the PDF, not just the truncated snippet.
    const snippets = (sources || [])
        .filter((s) => (s.source || "") === source)
        .map((s) => s.parent_content || s.content || s.snippet || "")
        .filter(Boolean);
    if (!snippets.length && focusSnippet) snippets.push(focusSnippet);

    citeState = {
        source, snippets, focusSnippet,
        view: "pdf",
        totalPages: null, currentPage: 0,
        pdfData: null, diffLoaded: false,
    };
    setCiteView("pdf");
}

// Switch the panel between the highlighted PDF and the workspace-vs-original
// Markdown diff for the same source. PDF page navigation only applies to the
// PDF view, so it is hidden in diff mode.
async function setCiteView(view) {
    if (!citeState) return;
    citeState.view = view;
    const panel = ensureCitePanel();
    panel.querySelectorAll(".cite-mode-btn").forEach((b) =>
        b.classList.toggle("is-active", b.dataset.mode === view));
    panel.querySelector(".cite-panel-nav").style.display = view === "pdf" ? "" : "none";
    if (view === "pdf") await renderCitePdfView();
    else await renderCiteDiffView();
}

async function renderCitePdfView() {
    const panel = ensureCitePanel();
    const body = panel.querySelector(".cite-panel-body");
    const loading = panel.querySelector(".cite-panel-loading");
    if (!citeState.pdfData) {
        body.innerHTML = "";
        loading.style.display = "block";
        const data = await fetchCiteDoc(citeState.source, citeState.snippets, citeState.focusSnippet);
        loading.style.display = "none";
        if (citeState.view !== "pdf") return;   // user toggled away while loading
        if (data.error || !Array.isArray(data.pages) || data.pages.length === 0) {
            body.innerHTML = `<div class="cite-panel-error"><span class="material-symbols-outlined">error</span> ${escapeHtml(data.error || "Could not load citation view.")}</div>`;
            return;
        }
        citeState.pdfData = data;
        citeState.totalPages = data.total_pages;
    }
    renderCiteDoc(citeState.pdfData);
}

async function renderCiteDiffView() {
    const panel = ensureCitePanel();
    const body = panel.querySelector(".cite-panel-body");
    const loading = panel.querySelector(".cite-panel-loading");
    body.onscroll = null;
    body.innerHTML = "";
    loading.style.display = "block";
    let orig = "", work = "";
    try {
        const src = encodeURIComponent(citeState.source);
        const [oRes, wRes] = await Promise.all([
            fetch(`${DOC_CONTENT_URL}/${src}/content?channel=originals`, { headers: authHeaders() }),
            fetch(`${DOC_CONTENT_URL}/${src}/content?channel=workspace`, { headers: authHeaders() }),
        ]);
        orig = oRes.ok ? ((await oRes.json()).markdown || "") : "";
        work = wRes.ok ? ((await wRes.json()).markdown || "") : "";
    } catch { /* fall through to error note below */ }
    loading.style.display = "none";
    if (citeState.view !== "diff") return;      // user toggled away while loading

    if (!work && !orig) {
        body.innerHTML = `<div class="cite-panel-error"><span class="material-symbols-outlined">error</span> No Markdown available for this document.</div>`;
        return;
    }
    const rows = diffLines(orig.split("\n"), work.split("\n"));
    const identical = orig.trim() === work.trim();
    let unified = "";
    rows.forEach((r) => {
        if (r.type === "same") {
            unified += `<div class="cite-diff-line same"><span class="cite-diff-sign"> </span>${escapeHtml(r.left ?? "") || "&nbsp;"}</div>`;
        } else if (r.type === "removed") {
            unified += `<div class="cite-diff-line removed"><span class="cite-diff-sign">−</span>${escapeHtml(r.left ?? "") || "&nbsp;"}</div>`;
        } else if (r.type === "added") {
            unified += `<div class="cite-diff-line added"><span class="cite-diff-sign">+</span>${escapeHtml(r.right ?? "") || "&nbsp;"}</div>`;
        }
    });
    body.innerHTML =
        (identical
            ? `<div class="cite-diff-note">No edits yet — workspace matches the original.</div>`
            : `<div class="cite-diff-note">Showing changes: <span class="cite-diff-legend removed">− original</span> <span class="cite-diff-legend added">+ workspace</span></div>`) +
        `<div class="cite-diff">${unified}</div>`;
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
            return { error: data.error || "Could not generate citation view." };
        }
        return data;
    } catch (e) {
        return { error: "Could not reach the server." };
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
        img.alt = `Page ${p.page}`;
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
        "I couldn't find enough information in the documents for this query. Would you like me to search online?";

    const actions = document.createElement("div");
    actions.className = "web-approval";

    const yesBtn = document.createElement("button");
    yesBtn.className = "web-approval-btn yes";
    yesBtn.innerHTML = `<span class="material-symbols-outlined">travel_explore</span> Yes, search online`;

    const noBtn = document.createElement("button");
    noBtn.className = "web-approval-btn no";
    noBtn.textContent = "No, that's fine";

    yesBtn.addEventListener("click", () => {
        actions.remove();
        logEvent("retrieval", "User approved web search.");
        runQuery(query, { allowWeb: true, echoUser: false });
    });
    noBtn.addEventListener("click", () => {
        actions.remove();
        contentDiv.textContent = "Understood, no web search performed.";
        logEvent("retrieval", "User declined web search.");
    });

    actions.appendChild(yesBtn);
    actions.appendChild(noBtn);
    botMessageDiv.appendChild(actions);
    logEvent("retrieval", "Not found in documents — awaiting user approval.");
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
    const time = new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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
        <p class="welcome-eyebrow">Hello 👋</p>
        <h1 class="welcome-title">What shall we explore today?</h1>
        <p class="welcome-lead">
            Type a question. Supervisor routes it, Researcher gathers evidence,
            Writer composes the answer, Reviewer does quality control — watch
            every step live on the right.
        </p>
        <div class="suggestion-row">
            <button class="suggestion" type="button">Summarize the main findings in the documents</button>
            <button class="suggestion" type="button">Compare two sources</button>
            <button class="suggestion" type="button">Extract a brief timeline</button>
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
    updateProgress(0, "Ready", "Idle");
    setStatus("Ready", "");
    logEvent("idle", "New session started.");
    userInput.focus();
});

/* ══════════════════════════════════════════════════════════════
   EDITOR TAB - RICH TEXT WYSIWYG & CHAT ACTIONS
   ═══════════════════════════════════════════════════════════════ */
let editorZoom = 1.0;

// Auto resize for editor composer
function autoResizeEditorInput() {
    editorUserInput.style.height = "auto";
    editorUserInput.style.height = `${editorUserInput.scrollHeight}px`;
}
editorUserInput.addEventListener("input", autoResizeEditorInput);
editorUserInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendReportChat();
    }
});

// Zoom controls
function updateZoom() {
    editorPage.style.setProperty("--editor-zoom", editorZoom);
    document.getElementById("editorZoomVal").textContent = `${Math.round(editorZoom * 100)}%`;
}

document.getElementById("editorZoomOut").addEventListener("click", () => {
    editorZoom = Math.max(0.5, editorZoom - 0.1);
    updateZoom();
});

document.getElementById("editorZoomIn").addEventListener("click", () => {
    editorZoom = Math.min(2.0, editorZoom + 0.1);
    updateZoom();
});

// toolbar data-cmd buttons
document.querySelectorAll("#editorToolbar .toolbar-btn[data-cmd]").forEach((btn) => {
    btn.addEventListener("mousedown", (e) => {
        e.preventDefault(); // maintain text selection
        const cmd = btn.dataset.cmd;
        document.execCommand(cmd, false, null);
        btn.classList.toggle("is-active", document.queryCommandState(cmd));
        triggerAutoSave();
    });
});

// Headings select
document.getElementById("editorHeadingSelect").addEventListener("change", (e) => {
    const tag = e.target.value;
    document.execCommand("formatBlock", false, tag);
    triggerAutoSave();
});

// Highlight Popover
const editorHighlightBtn = document.getElementById("editorHighlightBtn");
const editorHighlightPopover = document.getElementById("editorHighlightPopover");

editorHighlightBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    editorHighlightPopover.hidden = !editorHighlightPopover.hidden;
});

document.addEventListener("click", () => {
    if (editorHighlightPopover) editorHighlightPopover.hidden = true;
});

document.querySelectorAll(".color-swatch").forEach((swatch) => {
    swatch.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const color = swatch.dataset.color;
        if (color === "clear") {
            document.execCommand("removeFormat", false, null);
        } else {
            const colorMap = {
                yellow: "#fef08a",
                green: "#bbf7d0",
                blue: "#bfdbfe",
                red: "#fecaca"
            };
            document.execCommand("hiliteColor", false, colorMap[color]);
        }
        editorHighlightPopover.hidden = true;
        triggerAutoSave();
    });
});

// Add Chart action
document.getElementById("editorAddChartBtn").addEventListener("click", () => {
    editorUserInput.value = "@update add a chart to this section: [Table name]";
    editorUserInput.focus();
    autoResizeEditorInput();
});

// HTML ⇄ Markdown conversion
function htmlToMarkdown(container) {
    let md = "";
    for (const child of container.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
            const text = child.textContent.trim();
            if (text) md += text + "\n\n";
        } else if (child.nodeType === Node.ELEMENT_NODE) {
            const tagName = child.tagName.toLowerCase();
            if (/^h[1-6]$/.test(tagName)) {
                const level = parseInt(tagName[1]);
                md += "#".repeat(level) + " " + inlineHtmlToMarkdown(child) + "\n\n";
            } else if (tagName === "div" && child.classList.contains("kpi-card")) {
                md += child.outerHTML + "\n\n";
            } else if (tagName === "div" && child.classList.contains("page-break")) {
                md += child.outerHTML + "\n\n";
            } else if (tagName === "div" && child.classList.contains("report-chart-container")) {
                md += child.outerHTML + "\n\n";
            } else if (tagName === "p" || tagName === "div") {
                const text = inlineHtmlToMarkdown(child);
                if (text) md += text + "\n\n";
            } else if (tagName === "ul") {
                for (const li of child.querySelectorAll("li")) {
                    md += "- " + inlineHtmlToMarkdown(li) + "\n";
                }
                md += "\n";
            } else if (tagName === "ol") {
                let idx = 1;
                for (const li of child.querySelectorAll("li")) {
                    md += `${idx}. ` + inlineHtmlToMarkdown(li) + "\n";
                    idx++;
                }
                md += "\n";
            } else if (tagName === "blockquote") {
                const text = inlineHtmlToMarkdown(child);
                if (text) {
                    const lines = text.split("\n").map(l => "> " + l).join("\n");
                    md += lines + "\n\n";
                }
            } else if (tagName === "pre") {
                const code = child.querySelector("code");
                const text = code ? code.textContent : child.textContent;
                md += "```\n" + text.trim() + "\n```\n\n";
            } else if (tagName === "table") {
                const rows = child.querySelectorAll("tr");
                if (rows.length > 0) {
                    let tableMd = "";
                    rows.forEach((row, rIdx) => {
                        const cells = row.querySelectorAll("td, th");
                        const cellTexts = Array.from(cells).map(cell => inlineHtmlToMarkdown(cell).replace(/\|/g, "\\|"));
                        tableMd += "| " + cellTexts.join(" | ") + " |\n";
                        if (rIdx === 0) {
                            tableMd += "| " + cellTexts.map(() => "---").join(" | ") + " |\n";
                        }
                    });
                    md += tableMd + "\n";
                }
            } else if (tagName === "hr") {
                md += "---\n\n";
            } else {
                const text = inlineHtmlToMarkdown(child);
                if (text) md += text + "\n\n";
            }
        }
    }
    return md.trim();
}

function inlineHtmlToMarkdown(node) {
    let text = "";
    for (const child of node.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
            text += child.textContent;
        } else if (child.nodeType === Node.ELEMENT_NODE) {
            const tagName = child.tagName.toLowerCase();
            const inner = inlineHtmlToMarkdown(child);
            if (tagName === "br") {
                text += "\n";
            } else if (tagName === "strong" || tagName === "b") {
                text += `**${inner}**`;
            } else if (tagName === "em" || tagName === "i") {
                text += `*${inner}*`;
            } else if (tagName === "u") {
                text += `<u>${inner}</u>`;
            } else if (tagName === "mark") {
                text += `<mark>${inner}</mark>`;
            } else if (tagName === "del" || tagName === "strike" || tagName === "s") {
                text += `~~${inner}~~`;
            } else if (tagName === "code") {
                text += `\`${inner}\``;
            } else if (tagName === "a") {
                const href = child.getAttribute("href") || "";
                text += `[${inner}](${href})`;
            } else if (tagName === "img") {
                const alt = child.getAttribute("alt") || "";
                let src = child.getAttribute("src") || "";
                if (!src.startsWith("/images/") && src.includes("/")) {
                    const parts = src.split("/");
                    src = parts[parts.length - 1];
                }
                const style = child.getAttribute("style");
                const width = child.getAttribute("width");
                const height = child.getAttribute("height");
                if (style || width || height) {
                    let styleStr = style || "";
                    if (width && !styleStr.includes("width")) styleStr += ` width: ${width}px;`;
                    if (height && !styleStr.includes("height")) styleStr += ` height: ${height}px;`;
                    text += `<img src="${src}" alt="${alt}" style="${styleStr.trim()}">`;
                } else {
                    text += `![${alt}](${src})`;
                }
            } else {
                text += inner;
            }
        }
    }
    return text;
}

// Two-tier autosave: a fast per-keystroke draft write (markdown only) keeps
// editing responsive; a heavier "finalize" (reindex + PDF/DOCX + git commit)
// runs on idle / blur / before export so we don't reindex-and-commit on every
// keystroke burst.
let autoSaveTimer = null;      // fast draft-save debounce
let finalizeTimer = null;      // heavy finalize debounce
let hasUnfinalizedChanges = false;
const FINALIZE_IDLE_MS = 15000;

function triggerAutoSave() {
    if (!editorDoc) return;
    hasUnfinalizedChanges = true;
    setEditorSaveStatus("dirty");
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(saveDraft, 1200);
    if (finalizeTimer) clearTimeout(finalizeTimer);
    finalizeTimer = setTimeout(finalizeSave, FINALIZE_IDLE_MS);
}

async function saveDraft() {
    if (!editorDoc) return;
    setEditorSaveStatus("saving");
    const source = editorDoc.source;
    const currentMd = htmlToMarkdown(editorPage);
    try {
        const res = await fetch(`/documents/${encodeURIComponent(source)}/save-draft`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ markdown: currentMd })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || "Could not save.");
        }
        editorDoc.markdown = currentMd;
        setEditorSaveStatus("draft");
    } catch (err) {
        console.error("Draft save failed:", err);
        setEditorSaveStatus("error", err.message);
    }
}

// Heavy save: persist + reindex + regenerate PDF/DOCX + git commit.
async function finalizeSave() {
    if (!editorDoc) return;
    if (autoSaveTimer) { clearTimeout(autoSaveTimer); autoSaveTimer = null; }
    if (finalizeTimer) { clearTimeout(finalizeTimer); finalizeTimer = null; }
    if (!hasUnfinalizedChanges) return;
    setEditorSaveStatus("saving");
    const source = editorDoc.source;
    const currentMd = htmlToMarkdown(editorPage);
    try {
        const res = await fetch(`/documents/${encodeURIComponent(source)}/save`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ markdown: currentMd })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || "Could not save.");
        }
        editorDoc.markdown = currentMd;
        hasUnfinalizedChanges = false;
        setEditorSaveStatus("saved");
    } catch (err) {
        console.error("Finalize save failed:", err);
        setEditorSaveStatus("error", err.message);
    }
}

// Back-compat alias: existing callers (download, @update approve) want the heavy save.
async function runAutoSave() { await finalizeSave(); }

function setEditorSaveStatus(state, errMsg = "") {
    editorSaveStatus.className = `editor-save-pill ${state}`;
    const textEl = editorSaveStatus.querySelector(".status-text");
    editorSaveStatus.hidden = false;
    if (state === "saved") {
        textEl.textContent = "Saved & indexed";
        setTimeout(() => {
            if (editorSaveStatus.classList.contains("saved")) editorSaveStatus.hidden = true;
        }, 3000);
    } else if (state === "draft") {
        textEl.textContent = "Draft saved";
        setTimeout(() => {
            if (editorSaveStatus.classList.contains("draft")) editorSaveStatus.hidden = true;
        }, 2000);
    } else if (state === "saving") {
        textEl.textContent = "Saving…";
    } else if (state === "dirty") {
        textEl.textContent = "Unsaved changes";
    } else if (state === "error") {
        textEl.textContent = `Error: ${errMsg || "Could not save"}`;
    }
}

// Attach listeners for contenteditable changes
editorPage.addEventListener("input", () => {
    triggerAutoSave();
});
// Finalize (reindex/export/commit) when the user leaves the editor.
editorPage.addEventListener("blur", () => {
    if (hasUnfinalizedChanges) finalizeSave();
});

// Inline diff helpers
function _normHeading(text) {
    return text.replace(/^#+\s*/, "").replace(/\s*#+\s*$/, "").replace(/\s+/g, " ").trim().toLowerCase();
}

function getHeadingChain(headingEl, allHeadings) {
    const elLevel = parseInt(headingEl.tagName[1]);
    const chain = [headingEl];
    let currentLevel = elLevel;
    const idx = allHeadings.indexOf(headingEl);
    for (let i = idx - 1; i >= 0; i--) {
        const otherLevel = parseInt(allHeadings[i].tagName[1]);
        if (otherLevel < currentLevel) {
            chain.unshift(allHeadings[i]);
            currentLevel = otherLevel;
            if (otherLevel === 1) break;
        }
    }
    return chain.map(h => _normHeading(h.textContent));
}

function extractMarkdownSection(md, heading_path) {
    if (!heading_path) return md;
    const components = heading_path.split(">").map(_normHeading);
    const targetHeadingNorm = components[components.length - 1];
    const lines = md.split("\n");
    
    const headings = [];
    for (let i = 0; i < lines.length; i++) {
        const m = /^(#{1,6})\s+(.*)$/.exec(lines[i]);
        if (m) {
            headings.push({ idx: i, level: m[1].length, text: _normHeading(m[2]) });
        }
    }
    
    const candidates = headings.filter(h => h.text === targetHeadingNorm);
    if (!candidates.length) return null;
    
    let chosen = candidates[0];
    for (const cand of candidates) {
        const chain = [cand];
        let curLevel = cand.level;
        const candPos = headings.indexOf(cand);
        for (let j = candPos - 1; j >= 0; j--) {
            if (headings[j].level < curLevel) {
                chain.unshift(headings[j]);
                curLevel = headings[j].level;
                if (curLevel === 1) break;
            }
        }
        const chainTexts = chain.map(h => h.text);
        if (JSON.stringify(chainTexts) === JSON.stringify(components)) {
            chosen = cand;
            break;
        }
    }
    
    const startIdx = chosen.idx;
    let endIdx = lines.length;
    const candPos = headings.indexOf(chosen);
    for (let j = candPos + 1; j < headings.length; j++) {
        if (headings[j].level <= chosen.level) {
            endIdx = headings[j].idx;
            break;
        }
    }
    return lines.slice(startIdx, endIdx).join("\n");
}

function showInlineEditPreview(payload) {
    const heading_path = payload.heading_path || "";
    const headings = Array.from(editorPage.querySelectorAll("h1, h2, h3, h4, h5, h6"));
    const components = heading_path.split(">").map(_normHeading);
    const targetHeadingNorm = components[components.length - 1];
    
    let targetHeadingEl = null;
    for (const h of headings) {
        if (_normHeading(h.textContent) === targetHeadingNorm) {
            const chain = getHeadingChain(h, headings);
            if (JSON.stringify(chain) === JSON.stringify(components)) {
                targetHeadingEl = h;
                break;
            }
        }
    }
    if (!targetHeadingEl && targetHeadingNorm) {
        targetHeadingEl = headings.find(h => _normHeading(h.textContent) === targetHeadingNorm);
    }
    if (!targetHeadingEl) return false;
    
    const targetLevel = parseInt(targetHeadingEl.tagName[1]);
    const sectionNodes = [targetHeadingEl];
    let next = targetHeadingEl.nextElementSibling;
    while (next) {
        if (/^h[1-6]$/.test(next.tagName.toLowerCase())) {
            const nextLevel = parseInt(next.tagName[1]);
            if (nextLevel <= targetLevel) break;
        }
        sectionNodes.push(next);
        next = next.nextElementSibling;
    }
    
    const beforeSectionMd = extractMarkdownSection(payload.before, heading_path);
    const afterSectionMd = extractMarkdownSection(payload.after, heading_path);
    if (beforeSectionMd === null || afterSectionMd === null) return false;
    
    const diffWrapper = document.createElement("div");
    diffWrapper.className = "inline-diff-section";
    diffWrapper.contentEditable = "false";
    
    const diffRows = diffLines(beforeSectionMd.split("\n"), afterSectionMd.split("\n"));
    diffRows.forEach((row) => {
        const line = document.createElement("div");
        if (row.type === "added") {
            line.className = "inline-diff-line inline-diff-added";
            line.textContent = "+" + (row.right || "");
        } else if (row.type === "removed") {
            line.className = "inline-diff-line inline-diff-removed";
            line.textContent = "-" + (row.left || "");
        } else {
            line.className = "inline-diff-line inline-diff-same";
            line.textContent = row.right || "";
        }
        diffWrapper.appendChild(line);
    });
    
    const actions = document.createElement("div");
    actions.className = "inline-diff-actions";
    const approveBtn = document.createElement("button");
    approveBtn.type = "button";
    approveBtn.className = "primary-button";
    approveBtn.innerHTML = `<span class="material-symbols-outlined">check</span> Approve`;
    const rejectBtn = document.createElement("button");
    rejectBtn.type = "button";
    rejectBtn.className = "ghost-button";
    rejectBtn.innerHTML = `<span class="material-symbols-outlined">close</span> Reject`;
    actions.appendChild(rejectBtn);
    actions.appendChild(approveBtn);
    diffWrapper.appendChild(actions);
    
    const parent = targetHeadingEl.parentNode;
    parent.insertBefore(diffWrapper, targetHeadingEl);
    sectionNodes.forEach(node => parent.removeChild(node));
    
    approveBtn.addEventListener("click", async () => {
        approveBtn.disabled = true; rejectBtn.disabled = true;
        try {
            const res = await fetch("/update/apply", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ token: payload.token }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Approval failed.");

            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = renderMarkdown(afterSectionMd);
            while (tempDiv.firstChild) {
                diffWrapper.parentNode.insertBefore(tempDiv.firstChild, diffWrapper);
            }
            diffWrapper.parentNode.removeChild(diffWrapper);
            
            editorDoc.markdown = payload.after;
            setEditorSaveStatus("saved");
            
            addMessage(data.reply, "assistant");
            loadLibrary();
            loadEditorFileList();
        } catch (err) {
            alert(`Error: ${err.message}`);
            approveBtn.disabled = false; rejectBtn.disabled = false;
        }
    });
    
    rejectBtn.addEventListener("click", async () => {
        approveBtn.disabled = true; rejectBtn.disabled = true;
        try {
            await fetch("/update/reject", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ token: payload.token }),
            });
        } catch { /* ignored */ }
        
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = renderMarkdown(beforeSectionMd);
        while (tempDiv.firstChild) {
            diffWrapper.parentNode.insertBefore(tempDiv.firstChild, diffWrapper);
        }
        diffWrapper.parentNode.removeChild(diffWrapper);
        
        loadLibrary();
        loadEditorFileList();
    });
    
    return true;
}

// Show whole document diff for direct edit-chat suggestions
function showWholeDocumentDiff(beforeMd, afterMd) {
    editorPage.contentEditable = "false";
    
    // Clear and create a diff container inside the editor page
    editorPage.innerHTML = "";
    const diffContainer = document.createElement("div");
    diffContainer.className = "whole-document-diff";
    
    const diffRows = diffLines(beforeMd.split("\n"), afterMd.split("\n"));
    
    diffRows.forEach((row) => {
        const line = document.createElement("div");
        if (row.type === "added") {
            line.className = "inline-diff-line inline-diff-added";
            line.textContent = "+" + (row.right || "");
        } else if (row.type === "removed") {
            line.className = "inline-diff-line inline-diff-removed";
            line.textContent = "-" + (row.left || "");
        } else {
            line.className = "inline-diff-line inline-diff-same";
            line.textContent = row.right || "";
        }
        diffContainer.appendChild(line);
    });
    
    editorPage.appendChild(diffContainer);
    
    // Create floating actions bar at the bottom of the editor page wrapper
    let actions = document.getElementById("editorDiffActionsBar");
    if (!actions) {
        actions = document.createElement("div");
        actions.id = "editorDiffActionsBar";
        actions.className = "editor-diff-actions-bar";
        editorPage.parentNode.insertBefore(actions, editorPage.nextSibling);
    }
    
    actions.innerHTML = `
        <span class="diff-bar-text">AI has suggested changes. Review the changes:</span>
        <div class="diff-bar-buttons">
            <button class="primary-button" id="editorApproveDiffBtn" type="button"><span class="material-symbols-outlined">check</span> Accept Changes</button>
            <button class="ghost-button" id="editorRejectDiffBtn" type="button"><span class="material-symbols-outlined">close</span> Revert (Reject)</button>
        </div>
    `;
    actions.hidden = false;
    
    document.getElementById("editorApproveDiffBtn").onclick = async () => {
        editorDoc.markdown = afterMd;
        editorPage.innerHTML = renderMarkdown(afterMd);
        editorPage.contentEditable = "true";
        actions.hidden = true;

        hasUnfinalizedChanges = true;  // programmatic change → force a finalize
        await finalizeSave();
        loadLibrary();
        loadEditorFileList();
    };
    
    document.getElementById("editorRejectDiffBtn").onclick = () => {
        editorPage.innerHTML = renderMarkdown(beforeMd);
        editorPage.contentEditable = "true";
        actions.hidden = true;
    };
}

// Send editor message handler (direct edit-chat call)
/* ── Report studio chat (left rail): ask for evidence OR @update the report ── */
function addReportChatMsg(text, type) {
    const hint = reportChatMessages.querySelector(".report-chat-hint");
    if (hint) hint.remove();
    const el = document.createElement("div");
    el.className = `rc-msg ${type}`;
    if (type === "bot") el.innerHTML = renderMarkdown(text || "");
    else el.textContent = text;
    reportChatMessages.appendChild(el);
    reportChatMessages.scrollTop = reportChatMessages.scrollHeight;
    return el;
}

async function sendReportChat() {
    const message = editorUserInput.value.trim();
    if (!message || editorSendButton.disabled) return;

    const isEdit = /^@update\b/i.test(message) || /^@edit\b/i.test(message);

    editorUserInput.value = "";
    autoResizeEditorInput();
    addReportChatMsg(message, "user");
    editorUserInput.disabled = true;
    editorSendButton.disabled = true;

    try {
        if (isEdit) {
            await runReportEdit(message);
        } else {
            await runReportAsk(message);
        }
    } finally {
        editorUserInput.disabled = false;
        editorSendButton.disabled = false;
        editorUserInput.focus();
    }
}

// @update path: directly edit the open report and show the inline diff.
async function runReportEdit(message) {
    if (!editorDoc) {
        addReportChatMsg("Open or create a report first, then use @update to edit it.", "bot error");
        return;
    }
    const instruction = message.replace(/^@(update|edit)\b/i, "").trim() || message;
    const currentMd = htmlToMarkdown(editorPage);
    const ack = addReportChatMsg("_Computing changes…_", "bot");
    const originalContent = editorPage.innerHTML;
    editorPage.innerHTML = `
        <div class="editor-loading-overlay" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; gap: 16px; color: var(--muted);">
            <span class="material-symbols-outlined spin" style="font-size: 48px; animation: spin 2s linear infinite;">sync</span>
            <p style="font-size: 14px; font-weight: 500;">AI is computing changes, please wait…</p>
        </div>`;
    try {
        const res = await fetch(`/documents/${encodeURIComponent(editorDoc.source)}/edit-chat`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ query: instruction, current_markdown: currentMd })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || "Edit request failed.");
        }
        const data = await res.json();
        ack.innerHTML = renderMarkdown("Proposed changes are ready — review them in the report.");
        showWholeDocumentDiff(data.before, data.after);
    } catch (err) {
        ack.className = "rc-msg bot error";
        ack.textContent = err.message;
        editorPage.innerHTML = originalContent;
    }
}

// Ask path: run a RAG query, stream the answer into the rail, harvest evidence.
async function runReportAsk(message) {
    const bot = addReportChatMsg("", "bot");
    bot.classList.add("is-streaming");
    let fullText = "";
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
            body: JSON.stringify({ query: message, session_id: currentSessionId, allow_web: false, images: [] }),
        });
        if (!response.ok || !response.body) {
            let detail = "";
            try { const p = await response.json(); detail = p?.error ? ` ${p.error}` : ""; } catch {}
            throw new Error(`Request failed.${detail}`);
        }
        for await (const event of readSseEvents(response.body)) {
            if (event.event === "session_info") {
                const info = safeJson(event.data);
                if (info?.session_id) { currentSessionId = info.session_id; localStorage.setItem("rag_session_id", currentSessionId); }
            } else if (event.event === "search_results") {
                (safeJson(event.data) || []).forEach((r) => addEvidenceChip(r.content || "", r.source || r.title || ""));
            } else if (event.event === "token") {
                fullText += event.data;
                bot.innerHTML = renderMarkdown(fullText);
                reportChatMessages.scrollTop = reportChatMessages.scrollHeight;
            } else if (event.event === "message") {
                fullText = event.data;
                bot.innerHTML = renderMarkdown(fullText);
            } else if (event.event === "error") {
                const payload = safeJson(event.data);
                throw new Error(payload?.error || "Stream error.");
            }
        }
        if (fullText.trim()) addEvidenceChip(fullText, "Assistant answer");
        else bot.textContent = "No response was generated.";
    } catch (err) {
        bot.className = "rc-msg bot error";
        bot.textContent = `Error: ${err.message}`;
    } finally {
        bot.classList.remove("is-streaming");
        reportChatMessages.scrollTop = reportChatMessages.scrollHeight;
    }
}

editorChatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendReportChat();
});

/* ── Evidence chips ─────────────────────────────────────────── */
let evidenceItems = [];
function addEvidenceChip(text, source) {
    const clean = (text || "").trim();
    if (!clean) return;
    // De-dupe on the first 80 chars.
    const key = clean.slice(0, 80);
    if (evidenceItems.some((e) => e.key === key)) return;
    evidenceItems.unshift({ key, text: clean, source: source || "" });
    evidenceItems = evidenceItems.slice(0, 30);
    renderEvidence();
}

function renderEvidence() {
    if (evidenceCount) evidenceCount.textContent = evidenceItems.length ? String(evidenceItems.length) : "";
    if (!evidenceItems.length) {
        evidenceList.innerHTML = `<div class="empty-state small"><p>Ask a question in the chat — answers and sources land here to drag in.</p></div>`;
        return;
    }
    evidenceList.innerHTML = "";
    evidenceItems.forEach((item) => {
        const chip = document.createElement("div");
        chip.className = "evidence-chip";
        chip.draggable = true;
        chip.innerHTML = `<span class="ev-text">${escapeHtml(item.text)}</span>` +
            (item.source ? `<span class="ev-source">${escapeHtml(item.source)}</span>` : "");
        chip.addEventListener("dragstart", (e) => {
            e.dataTransfer.setData("application/x-report-evidence", JSON.stringify(item));
            e.dataTransfer.effectAllowed = "copy";
        });
        chip.addEventListener("click", () => insertEvidence(item));
        evidenceList.appendChild(chip);
    });
}

function evidenceToHtml(item) {
    const src = item.source ? ` <cite>— ${escapeHtml(item.source)}</cite>` : "";
    // Keep it short for a citation blockquote.
    const snippet = item.text.length > 600 ? item.text.slice(0, 600) + "…" : item.text;
    return `<blockquote>${escapeHtml(snippet)}${src}</blockquote>`;
}
function insertEvidence(item) { insertHtmlIntoReport(evidenceToHtml(item)); }

/* ── Library assets (figures & tables) ──────────────────────── */
async function loadLibraryAssets() {
    if (!libraryAssets) return;
    try {
        const res = await fetch("/library/assets", { headers: authHeaders() });
        if (!res.ok) throw new Error("failed");
        const data = await res.json();
        renderLibraryAssets(data.figures || [], data.tables || []);
    } catch {
        libraryAssets.innerHTML = `<div class="empty-state small"><p>Could not load library assets.</p></div>`;
    }
}

function renderLibraryAssets(figures, tables) {
    populateChartTables(tables);
    libraryAssets.innerHTML = "";
    if (!figures.length && !tables.length) {
        libraryAssets.innerHTML = `<div class="empty-state small"><p>No figures or tables in the library yet.</p></div>`;
    }
    if (figures.length) {
        const g = document.createElement("div");
        g.innerHTML = `<div class="asset-group-title">Figures</div>`;
        const grid = document.createElement("div");
        grid.className = "asset-figures";
        figures.forEach((f) => {
            const cell = document.createElement("div");
            cell.className = "asset-fig";
            cell.draggable = true;
            cell.title = `${f.name} (${f.source})`;
            cell.innerHTML = `<img src="${escapeHtml(f.url)}" alt="${escapeHtml(f.name)}" loading="lazy">`;
            cell.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData("application/x-report-figure", JSON.stringify(f));
                e.dataTransfer.effectAllowed = "copy";
            });
            cell.addEventListener("click", () => insertFigure(f));
            grid.appendChild(cell);
        });
        g.appendChild(grid);
        libraryAssets.appendChild(g);
    }
    if (tables.length) {
        const g = document.createElement("div");
        g.innerHTML = `<div class="asset-group-title">Tables</div>`;
        tables.forEach((t) => {
            const row = document.createElement("div");
            row.className = "asset-table";
            row.draggable = true;
            row.innerHTML = `<span class="material-symbols-outlined">table</span><span>${escapeHtml(t.name)} · ${t.row_count} rows</span>`;
            row.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData("application/x-report-table", JSON.stringify(t));
                e.dataTransfer.effectAllowed = "copy";
            });
            row.addEventListener("click", () => insertLibraryTable(t));
            g.appendChild(row);
        });
        libraryAssets.appendChild(g);
    }
}

async function insertFigure(f) {
    if (!editorDoc) { alert("Open or create a report first."); return; }
    // Copy the library figure into this report's images so exports resolve it.
    try {
        const imgRes = await fetch(f.url);
        const blob = await imgRes.blob();
        const dataUrl = await blobToDataUrl(blob);
        const up = await fetch(`/documents/${encodeURIComponent(editorDoc.source)}/images`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ data_url: dataUrl, name: f.name })
        });
        const data = await up.json();
        if (!up.ok) throw new Error(data.error || "upload failed");
        insertHtmlIntoReport(`<p><img src="${escapeHtml(data.url)}" alt="${escapeHtml(f.name)}" class="report-img"></p>`);
    } catch (err) {
        alert(`Could not add figure: ${err.message}`);
    }
}

function insertLibraryTable(t) {
    const headers = t.headers || [];
    const rows = t.rows_preview || [];
    let html = "<table><thead><tr>";
    headers.forEach((h) => { html += `<th>${escapeHtml(String(h))}</th>`; });
    html += "</tr></thead><tbody>";
    rows.forEach((r) => {
        html += "<tr>";
        (Array.isArray(r) ? r : headers.map((h) => r[h] ?? "")).forEach((c) => {
            html += `<td>${escapeHtml(String(c ?? ""))}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    insertHtmlIntoReport(html);
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = reject;
        fr.readAsDataURL(blob);
    });
}

/* ── Version history ────────────────────────────────────────── */
if (editorHistoryBtn) editorHistoryBtn.addEventListener("click", openHistory);
if (historyClose) historyClose.addEventListener("click", () => { historyModal.hidden = true; });
if (historyModal) historyModal.addEventListener("click", (e) => { if (e.target === historyModal) historyModal.hidden = true; });

async function openHistory() {
    if (!editorDoc) return;
    historyModal.hidden = false;
    historySub.textContent = editorDoc.title || editorDoc.source;
    historyList.innerHTML = `<div class="empty-state small"><p>Loading history…</p></div>`;
    try {
        const res = await fetch(`/admin/history?source=${encodeURIComponent(editorDoc.source)}&limit=50`, { headers: authHeaders() });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Could not load history.");
        renderHistory(data.commits || []);
    } catch (err) {
        historyList.innerHTML = `<div class="empty-state small"><p>${escapeHtml(err.message)}</p></div>`;
    }
}

function renderHistory(commits) {
    if (!commits.length) {
        historyList.innerHTML = `<div class="empty-state small"><p>No saved versions yet. Edits are versioned when the report is finalized (on idle or download).</p></div>`;
        return;
    }
    historyList.innerHTML = "";
    commits.forEach((c, i) => {
        const when = c.date ? new Date(c.date).toLocaleString() : "";
        const row = document.createElement("div");
        row.className = "history-row";
        row.innerHTML = `
            <div class="history-meta">
                <span class="history-msg">${escapeHtml((c.message || "").replace(/^@update\s+\S+:\s*/, ""))}</span>
                <span class="history-when">${escapeHtml(when)} · ${escapeHtml(c.short_sha || "")}</span>
            </div>
            ${i === 0 ? `<span class="history-current">current</span>`
                     : `<button type="button" class="ghost-button history-restore" data-ref="${escapeHtml(c.sha)}">Restore</button>`}`;
        historyList.appendChild(row);
    });
    historyList.querySelectorAll(".history-restore").forEach((b) => {
        b.addEventListener("click", () => restoreVersion(b.dataset.ref));
    });
}

async function restoreVersion(ref) {
    if (!editorDoc) return;
    if (!window.confirm("Restore this version? Current unsaved changes will be replaced.")) return;
    try {
        const res = await fetch("/admin/restore", {
            method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ source: editorDoc.source, ref })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Restore failed.");
        historyModal.hidden = true;
        hasUnfinalizedChanges = false;
        await selectEditorDoc(editorDoc.source);
    } catch (err) {
        alert(err.message);
    }
}

// Flush any pending edits, then open the freshly-regenerated export.
async function downloadReport(kind) {
    if (!editorDoc) return;
    let btn = null;
    if (kind === "pdf") btn = editorDownloadPdf;
    else if (kind === "docx") btn = editorDownloadDocx;
    else if (kind === "markdown") btn = editorDownloadMd;
    else if (kind === "html") btn = editorDownloadHtml;

    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    try {
        if (btn) btn.disabled = true;
        await runAutoSave();
    } catch {}
    finally { if (btn) btn.disabled = false; }
    window.open(`/workspace/${kind}/${encodeURIComponent(editorDoc.source)}`, "_blank");
}

/* ── New report modal ───────────────────────────────────────── */
let selectedTemplate = "blank";
function openNewReportModal() {
    if (!isAuthenticated()) { openAuthModal(); return; }
    newReportTitle.value = "";
    if (newReportAIPrompt) newReportAIPrompt.value = "";
    if (newReportUseAI) {
        newReportUseAI.checked = false;
        if (aiGenPromptWrap) aiGenPromptWrap.hidden = true;
        if (aiGenStatusWrap) aiGenStatusWrap.hidden = true;
        if (newReportActions) newReportActions.hidden = false;
        newReportCreate.innerHTML = `<span class="material-symbols-outlined">note_add</span> Create report`;
    }
    newReportStatus.hidden = true;
    newReportStatus.className = "new-report-status";
    selectedTemplate = "blank";
    templateGrid.querySelectorAll(".template-card").forEach((c) =>
        c.classList.toggle("is-active", c.dataset.template === "blank"));
    newReportModal.hidden = false;
    setTimeout(() => newReportTitle.focus(), 30);
}
function closeNewReportModal() { newReportModal.hidden = true; }

if (newReportBtn) newReportBtn.addEventListener("click", openNewReportModal);
if (newReportClose) newReportClose.addEventListener("click", closeNewReportModal);
if (newReportModal) newReportModal.addEventListener("click", (e) => { if (e.target === newReportModal) closeNewReportModal(); });
if (templateGrid) templateGrid.addEventListener("click", (e) => {
    const card = e.target.closest(".template-card");
    if (!card) return;
    selectedTemplate = card.dataset.template;
    templateGrid.querySelectorAll(".template-card").forEach((c) => c.classList.toggle("is-active", c === card));
});
if (newReportUseAI) {
    newReportUseAI.addEventListener("change", () => {
        if (aiGenPromptWrap) aiGenPromptWrap.hidden = !newReportUseAI.checked;
        if (newReportCreate) {
            newReportCreate.innerHTML = newReportUseAI.checked
                ? `<span class="material-symbols-outlined">auto_awesome</span> Generate with AI`
                : `<span class="material-symbols-outlined">note_add</span> Create report`;
        }
    });
}
if (newReportCreate) newReportCreate.addEventListener("click", createNewReport);

async function createNewReport() {
    const title = newReportTitle.value.trim();
    if (!title) {
        newReportStatus.textContent = "Please enter a title.";
        newReportStatus.className = "new-report-status error";
        newReportStatus.hidden = false;
        return;
    }
    
    if (newReportUseAI && newReportUseAI.checked) {
        const topic = (newReportAIPrompt ? newReportAIPrompt.value.trim() : "");
        if (!topic) {
            newReportStatus.textContent = "Please enter report topic/instructions.";
            newReportStatus.className = "new-report-status error";
            newReportStatus.hidden = false;
            return;
        }
        if (aiGenerationSubmitting) return;
        aiGenerationSubmitting = true;
        newReportCreate.disabled = true;
        const opts = {
            length: (document.getElementById("aiGenLength") || {}).value || "standard",
            language: (document.getElementById("aiGenLanguage") || {}).value || "auto",
            tone: (document.getElementById("aiGenTone") || {}).value || "",
            audience: ((document.getElementById("aiGenAudience") || {}).value || "").trim(),
        };
        try {
            await createNewReportAI(title, topic, selectedTemplate, opts);
        } finally {
            aiGenerationSubmitting = false;
            newReportCreate.disabled = false;
        }
        return;
    }

    newReportCreate.disabled = true;
    newReportStatus.textContent = "Creating…";
    newReportStatus.className = "new-report-status";
    newReportStatus.hidden = false;
    try {
        const res = await fetch("/documents/create", {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ title, template: selectedTemplate })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            if (res.status === 404) throw new Error("Endpoint /documents/create not found — the backend needs restarting/redeploying with the latest code.");
            throw new Error(data.error || `Creation failed (HTTP ${res.status}).`);
        }
        closeNewReportModal();
        await loadEditorFileList();
        await selectEditorDoc(data.source);
    } catch (err) {
        newReportStatus.textContent = err.message;
        newReportStatus.className = "new-report-status error";
    } finally {
        newReportCreate.disabled = false;
    }
}

let aiGenerationSubmitting = false;

/* ── Toast notifications ─────────────────────────────────────── */
const toastContainer = document.getElementById("toastContainer");

function showToast(message, { type = "success", onClick = null, duration = 6000 } = {}) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}${onClick ? " toast-clickable" : ""}`;
    const icon = type === "error" ? "error" : "check_circle";
    toast.innerHTML = `<span class="material-symbols-outlined">${icon}</span><span>${escapeHtml(message)}</span>`;

    const dismiss = () => {
        toast.classList.remove("is-visible");
        setTimeout(() => toast.remove(), 200);
    };

    if (onClick) {
        toast.addEventListener("click", () => { onClick(); dismiss(); });
    }

    toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("is-visible"));
    setTimeout(dismiss, duration);
}

async function createNewReportAI(title, topic, template, opts = {}) {
    let res;
    try {
        res = await fetch("/documents/generate", {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({
                title, topic, template,
                length: opts.length || "standard",
                language: opts.language || "auto",
                tone: opts.tone || "",
                audience: opts.audience || "",
            })
        });
    } catch (err) {
        newReportStatus.textContent = err.message || "Could not reach the server.";
        newReportStatus.className = "new-report-status error";
        newReportStatus.hidden = false;
        return;
    }

    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        newReportStatus.textContent = data.error || `Generation failed (HTTP ${res.status}).`;
        newReportStatus.className = "new-report-status error";
        newReportStatus.hidden = false;
        return;
    }

    // Request accepted and the graph is running server-side. Switch the modal
    // into live-progress mode: hide the prompt/actions, reveal the progress
    // panel, and stream the graph's status / section events into it. The user
    // may close the modal to keep it running in the background (a toast reports
    // the outcome), but by default they watch it build section by section.
    if (aiGenPromptWrap) aiGenPromptWrap.hidden = true;
    if (newReportActions) newReportActions.hidden = true;
    if (newReportStatus) newReportStatus.hidden = true;
    _resetAiGenProgress();
    if (aiGenStatusWrap) aiGenStatusWrap.hidden = false;

    await runReportGeneration(res, title);
}

/* ── AI report generation progress helpers ──────────────────────── */
function _resetAiGenProgress() {
    if (aiGenStatusText) aiGenStatusText.textContent = "Planning outline…";
    if (aiGenProgressFill) aiGenProgressFill.style.width = "0%";
    if (aiGenLogs) aiGenLogs.innerHTML = "";
}

function _appendAiGenLog(message) {
    if (!aiGenLogs) return;
    const line = document.createElement("div");
    line.className = "ai-gen-log-line";
    line.textContent = message;
    aiGenLogs.appendChild(line);
    aiGenLogs.scrollTop = aiGenLogs.scrollHeight;
}

function _setAiGenProgress(done, total) {
    if (!aiGenProgressFill || !total) return;
    // Cap at 95% until the `complete` event so the bar never reads "done" early.
    const pct = Math.min(95, Math.round((Math.min(done, total) / total) * 100));
    aiGenProgressFill.style.width = `${pct}%`;
}

async function runReportGeneration(res, title) {
    let sectionsTotal = 0;
    let sectionsDone = 0;
    let finished = false;
    try {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            let parts = buffer.split("\n\n");
            buffer = parts.pop(); // Keep the last incomplete part

            for (const part of parts) {
                if (!part.trim()) continue;

                let event = "message";
                let data = "";

                const lines = part.split("\n");
                for (const line of lines) {
                    if (line.startsWith("event: ")) {
                        event = line.slice(7).trim();
                    } else if (line.startsWith("data: ")) {
                        data = line.slice(6).trim();
                    }
                }

                let payload = {};
                try { payload = data ? JSON.parse(data) : {}; } catch { payload = {}; }

                if (event === "status") {
                    if (payload.message && aiGenStatusText) aiGenStatusText.textContent = payload.message;
                    if (payload.message) _appendAiGenLog(payload.message);
                } else if (event === "section_start") {
                    sectionsTotal += 1;
                    if (payload.title) _appendAiGenLog(`• ${payload.title}`);
                    if (aiGenStatusText) aiGenStatusText.textContent = `Outline: ${sectionsTotal} section(s)…`;
                } else if (event === "section_complete") {
                    sectionsDone += 1;
                    if (payload.title) _appendAiGenLog(`✓ ${payload.title}`);
                    _setAiGenProgress(sectionsDone, sectionsTotal);
                    if (aiGenStatusText) {
                        aiGenStatusText.textContent = `Writing sections… (${Math.min(sectionsDone, sectionsTotal)}/${sectionsTotal})`;
                    }
                } else if (event === "complete") {
                    finished = true;
                    if (aiGenProgressFill) aiGenProgressFill.style.width = "100%";
                    if (aiGenStatusText) aiGenStatusText.textContent = "Report ready.";
                    closeNewReportModal();
                    _resetNewReportModalUI();
                    showToast(`Report generated: "${title}"`, {
                        type: "success",
                        onClick: () => selectEditorDoc(payload.source),
                    });
                    await loadEditorFileList();
                    if (payload.source) await selectEditorDoc(payload.source);
                    return;
                } else if (event === "error") {
                    throw new Error(payload.error || "Generation error");
                }
            }
        }
        if (!finished) throw new Error("Stream ended before the report was ready.");
    } catch (err) {
        // If the modal is still open, surface the error inline; otherwise toast.
        if (newReportModal && !newReportModal.hidden) {
            if (aiGenStatusWrap) aiGenStatusWrap.hidden = true;
            if (newReportActions) newReportActions.hidden = false;
            if (newReportStatus) {
                newReportStatus.textContent = `Generation failed — ${err.message}`;
                newReportStatus.className = "new-report-status error";
                newReportStatus.hidden = false;
            }
        }
        showToast(`Report generation failed: "${title}" — ${err.message}`, { type: "error", duration: 8000 });
    }
}

function _resetNewReportModalUI() {
    if (newReportUseAI) newReportUseAI.checked = false;
    if (aiGenPromptWrap) aiGenPromptWrap.hidden = true;
    if (aiGenStatusWrap) aiGenStatusWrap.hidden = true;
    if (newReportActions) newReportActions.hidden = false;
    if (newReportStatus) newReportStatus.hidden = true;
    if (newReportCreate) {
        newReportCreate.innerHTML = `<span class="material-symbols-outlined">note_add</span> Create report`;
    }
}

/* ── Tools panel toggle ─────────────────────────────────────── */
if (reportToolsToggleBtn) reportToolsToggleBtn.addEventListener("click", () => {
    // On wide screens collapse the column; on narrow screens toggle the overlay.
    if (window.matchMedia("(max-width: 1180px)").matches) {
        editorView.classList.toggle("report-tools-open");
    } else {
        editorView.classList.toggle("report-tools-collapsed");
    }
});

/* ══════════════════════════════════════════════════════════════
   Insert-into-report core + content blocks + drag & drop
═══════════════════════════════════════════════════════════════ */

// Insert an HTML fragment into the report page at the caret (or append), then save.
function insertHtmlIntoReport(html, dropRange = null) {
    if (!editorDoc) { alert("Open or create a report first."); return; }
    if (editorPage.hidden) return;
    editorPage.focus();

    const sel = window.getSelection();
    let range = dropRange;
    if (!range) {
        if (sel && sel.rangeCount && editorPage.contains(sel.anchorNode)) {
            range = sel.getRangeAt(0);
        }
    }

    const temp = document.createElement("div");
    temp.innerHTML = html;
    const frag = document.createDocumentFragment();
    let lastNode = null;
    while (temp.firstChild) { lastNode = temp.firstChild; frag.appendChild(lastNode); }

    if (range && editorPage.contains(range.startContainer)) {
        range.collapse(false);
        range.insertNode(frag);
        if (lastNode) {
            const after = document.createRange();
            after.setStartAfter(lastNode);
            after.collapse(true);
            sel.removeAllRanges();
            sel.addRange(after);
        }
    } else {
        editorPage.appendChild(frag);
    }
    triggerAutoSave();
}

const BLOCK_HTML = {
    h1: "<h1>Heading</h1>",
    h2: "<h2>Heading</h2>",
    h3: "<h3>Heading</h3>",
    p: "<p>New paragraph. Click to edit.</p>",
    ul: "<ul><li>First item</li><li>Second item</li></ul>",
    ol: "<ol><li>First item</li><li>Second item</li></ol>",
    quote: "<blockquote>Quote or callout text.</blockquote>",
    divider: "<hr>",
    pagebreak: '<div class="page-break" style="page-break-after: always; border-top: 1px dashed var(--border); margin: 18px 0;"></div>',
    kpi: '<div class="kpi-card"><span class="kpi-value">0</span><span class="kpi-label">Metric</span></div>',
};

function blockHtml(type) {
    if (type === "table") {
        let h = "<table><thead><tr><th>Column A</th><th>Column B</th><th>Column C</th></tr></thead><tbody>";
        for (let i = 0; i < 3; i++) h += "<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>";
        return h + "</tbody></table>";
    }
    return BLOCK_HTML[type] || "";
}

// Content-block buttons: click to insert, drag to place.
if (blockGrid) {
    blockGrid.querySelectorAll(".block-item").forEach((btn) => {
        btn.addEventListener("click", () => insertHtmlIntoReport(blockHtml(btn.dataset.block)));
        btn.addEventListener("dragstart", (e) => {
            e.dataTransfer.setData("application/x-report-block", btn.dataset.block);
            e.dataTransfer.effectAllowed = "copy";
        });
    });
}

// The report page is a drop target for blocks, evidence, figures and tables.
editorPage.addEventListener("dragover", (e) => {
    if (editorPage.hidden) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    editorPage.classList.add("drop-active");
});
editorPage.addEventListener("dragleave", () => editorPage.classList.remove("drop-active"));
editorPage.addEventListener("drop", (e) => {
    editorPage.classList.remove("drop-active");
    if (editorPage.hidden) return;
    const dt = e.dataTransfer;
    let html = null;
    const block = dt.getData("application/x-report-block");
    const ev = dt.getData("application/x-report-evidence");
    const fig = dt.getData("application/x-report-figure");
    const tbl = dt.getData("application/x-report-table");
    if (block) html = blockHtml(block);
    else if (ev) { try { insertEvidence(JSON.parse(ev)); } catch {} return e.preventDefault(); }
    else if (fig) { try { insertFigure(JSON.parse(fig)); } catch {} return e.preventDefault(); }
    else if (tbl) { try { insertLibraryTable(JSON.parse(tbl)); } catch {} return e.preventDefault(); }
    if (!html) return;
    e.preventDefault();
    let range = null;
    if (document.caretRangeFromPoint) range = document.caretRangeFromPoint(e.clientX, e.clientY);
    else if (document.caretPositionFromPoint) {
        const pos = document.caretPositionFromPoint(e.clientX, e.clientY);
        if (pos) { range = document.createRange(); range.setStart(pos.offsetNode, pos.offset); }
    }
    insertHtmlIntoReport(html, range);
});

/* ══════════════════════════════════════════════════════════════
   Chart builder (Chart.js) → PNG → workspace image → report
═══════════════════════════════════════════════════════════════ */
const chartTypeEl = document.getElementById("chartType");
const chartTitleEl = document.getElementById("chartTitle");
const chartDataEl = document.getElementById("chartData");
const chartTableSelect = document.getElementById("chartTableSelect");
const chartPreviewCanvas = document.getElementById("chartPreview");
const chartInsertBtn = document.getElementById("chartInsertBtn");
const chartAiBtn = document.getElementById("chartAiBtn");
let chartPreviewInstance = null;
let chartTablesCache = [];

function parseChartData(text) {
    const labels = [], values = [];
    (text || "").split("\n").forEach((line) => {
        const t = line.trim();
        if (!t) return;
        const idx = t.lastIndexOf(",");
        if (idx === -1) return;
        const label = t.slice(0, idx).trim();
        const val = parseFloat(t.slice(idx + 1).trim().replace(/[^0-9.\-]/g, ""));
        if (label && !isNaN(val)) { labels.push(label); values.push(val); }
    });
    return { labels, values };
}

const CHART_PALETTE = ["#c96442", "#e0a458", "#6b8f71", "#4f7cac", "#9b6a9e", "#c9a94b", "#7a9e9f", "#b5654a"];

function buildChartConfig() {
    const type = chartTypeEl ? chartTypeEl.value : "bar";
    const title = chartTitleEl ? chartTitleEl.value.trim() : "";
    const { labels, values } = parseChartData(chartDataEl ? chartDataEl.value : "");
    const multi = ["pie", "doughnut", "polarArea"].includes(type);
    return {
        type,
        data: {
            labels,
            datasets: [{
                label: title || "Series",
                data: values,
                backgroundColor: multi ? labels.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]) : "#c96442",
                borderColor: multi ? "#fff" : "#c96442",
                borderWidth: multi ? 2 : 1,
                fill: type === "line" ? false : true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: multi, position: "bottom" },
                title: { display: Boolean(title), text: title },
            },
            scales: (type === "bar" || type === "line") ? { y: { beginAtZero: true } } : {},
        },
    };
}

function renderChartPreview() {
    if (typeof Chart === "undefined" || !chartPreviewCanvas) return;
    const wrap = chartPreviewCanvas.parentElement;
    const cfg = buildChartConfig();
    if (!cfg.data.labels.length) {
        if (chartPreviewInstance) { chartPreviewInstance.destroy(); chartPreviewInstance = null; }
        if (wrap) wrap.classList.remove("has-chart");
        return;
    }
    if (chartPreviewInstance) chartPreviewInstance.destroy();
    chartPreviewInstance = new Chart(chartPreviewCanvas, cfg);
    if (wrap) wrap.classList.add("has-chart");
}

[chartTypeEl, chartTitleEl, chartDataEl].forEach((el) => {
    if (el) el.addEventListener("input", () => renderChartPreview());
});

if (chartTableSelect) chartTableSelect.addEventListener("change", () => {
    const idx = chartTableSelect.value;
    if (idx === "") return;
    const t = chartTablesCache[parseInt(idx)];
    if (!t) return;
    // Use the first column as label and the first numeric column as value.
    const headers = t.headers || [];
    const rows = t.rows_preview || [];
    let numCol = 1;
    if (rows.length) {
        for (let c = 1; c < headers.length; c++) {
            const v = Array.isArray(rows[0]) ? rows[0][c] : rows[0][headers[c]];
            if (!isNaN(parseFloat(v))) { numCol = c; break; }
        }
    }
    const lines = rows.map((r) => {
        const label = Array.isArray(r) ? r[0] : r[headers[0]];
        const val = Array.isArray(r) ? r[numCol] : r[headers[numCol]];
        return `${label}, ${val}`;
    });
    if (chartDataEl) chartDataEl.value = lines.join("\n");
    if (chartTitleEl && !chartTitleEl.value) chartTitleEl.value = headers[numCol] || "";
    renderChartPreview();
});

// Populate the "from table" dropdown when assets load.
function populateChartTables(tables) {
    chartTablesCache = tables || [];
    if (!chartTableSelect) return;
    chartTableSelect.innerHTML = `<option value="">— manual entry —</option>`;
    chartTablesCache.forEach((t, i) => {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${t.name} (${t.source})`;
        chartTableSelect.appendChild(opt);
    });
}

let currentEditingChartContainer = null;

if (chartInsertBtn) chartInsertBtn.addEventListener("click", async () => {
    if (!editorDoc) { alert("Open or create a report first."); return; }
    if (typeof Chart === "undefined") { alert("Chart library is still loading — try again in a moment."); return; }
    const cfg = buildChartConfig();
    if (!cfg.data.labels.length) { alert("Add some data first (one 'label, value' per line)."); return; }

    // Render offscreen on a white background for clean export.
    const off = document.createElement("canvas");
    off.width = 900; off.height = 520;
    document.body.appendChild(off);
    const ctx = off.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, off.width, off.height);
    const inst = new Chart(off, cfg);
    // Chart.js draws synchronously with animation:false; give one frame just in case.
    await new Promise((r) => requestAnimationFrame(r));
    const dataUrl = off.toDataURL("image/png");
    inst.destroy();
    off.remove();

    chartInsertBtn.disabled = true;
    try {
        const res = await fetch(`/documents/${encodeURIComponent(editorDoc.source)}/images`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ data_url: dataUrl, name: `chart_${Date.now()}` })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Chart upload failed.");
        const alt = (chartTitleEl && chartTitleEl.value.trim()) || "chart";
        
        const spec = {
            type: cfg.type,
            title: cfg.options.plugins.title.text || "",
            labels: cfg.data.labels,
            values: cfg.data.datasets[0].data
        };

        if (currentEditingChartContainer) {
            const img = currentEditingChartContainer.querySelector("img");
            if (img) {
                img.src = data.url;
                img.alt = alt;
            }
            currentEditingChartContainer.dataset.spec = JSON.stringify(spec);
            currentEditingChartContainer = null;
            chartInsertBtn.innerHTML = `<span class="material-symbols-outlined">insert_chart</span> Insert chart`;
            triggerAutoSave();
        } else {
            const htmlBlock = `<div class="report-chart-container" data-spec="${escapeHtml(JSON.stringify(spec))}" contenteditable="false"><img src="${escapeHtml(data.url)}" alt="${escapeHtml(alt)}" class="report-img report-chart"></div>`;
            insertHtmlIntoReport(htmlBlock);
        }
    } catch (err) {
        alert(err.message);
    } finally {
        chartInsertBtn.disabled = false;
    }
});

// "Ask AI": route the request through the existing table-driven chart path via @update.
if (chartAiBtn) chartAiBtn.addEventListener("click", () => {
    const type = chartTypeEl ? chartTypeEl.value : "bar";
    const src = chartTableSelect && chartTableSelect.value !== ""
        ? (chartTablesCache[parseInt(chartTableSelect.value)] || {}).name || ""
        : "";
    editorUserInput.value = `@update add a ${type} chart${src ? ` from the ${src} table` : ""} to this report`;
    editorUserInput.focus();
    autoResizeEditorInput();
});

async function selectEditorDoc(source) {
    // Flush any pending finalize for the doc we're leaving, then reset save state.
    if (hasUnfinalizedChanges) { try { await finalizeSave(); } catch {} }
    if (autoSaveTimer) { clearTimeout(autoSaveTimer); autoSaveTimer = null; }
    if (finalizeTimer) { clearTimeout(finalizeTimer); finalizeTimer = null; }
    hasUnfinalizedChanges = false;

    editorDoc = null;
    editorPage.hidden = true;
    editorDownloadDocx.disabled = true;
    if (editorDownloadPdf) editorDownloadPdf.disabled = true;
    editorFileList.querySelectorAll(".library-row").forEach((r) =>
        r.classList.toggle("is-active", r.dataset.source === source));

    try {
        const res = await fetch(`${DOC_CONTENT_URL}/${encodeURIComponent(source)}/content?channel=workspace`);
        if (!res.ok) throw new Error("Load failed");
        const data = await res.json();
        
        editorDoc = {
            source,
            markdown: data.markdown || ""
        };

        // Render HTML
        editorPage.innerHTML = renderMarkdown(editorDoc.markdown);
        editorPage.hidden = false;
        
        // Enable download buttons — flush pending edits first so exports are current.
        editorDownloadDocx.disabled = false;
        editorDownloadDocx.onclick = () => downloadReport("docx");
        if (editorDownloadPdf) {
            editorDownloadPdf.disabled = false;
            editorDownloadPdf.onclick = () => downloadReport("pdf");
        }
        if (editorDownloadMd) {
            editorDownloadMd.disabled = false;
            editorDownloadMd.onclick = () => downloadReport("markdown");
        }
        if (editorDownloadHtml) {
            editorDownloadHtml.disabled = false;
            editorDownloadHtml.onclick = () => downloadReport("html");
        }
        if (editorHistoryBtn) editorHistoryBtn.disabled = false;

        // Refresh the tools panel's library assets for this workspace
        loadLibraryAssets();

        // Hide save status initially
        setEditorSaveStatus("saved");

        // Rebuild Outline
        rebuildOutline();
    } catch (err) {
        console.error("selectEditorDoc error:", err);
        editorPage.innerHTML = `<p style="color: var(--danger); padding: 20px;">Could not load document.</p>`;
        editorPage.hidden = false;
    }
}

/* ── Selection floating toolbar AI assistant ──────────────────── */
const editorSelectionToolbar = document.getElementById("editorSelectionToolbar");
const stCustomPrompt = document.getElementById("stCustomPrompt");
const stCustomSend = document.getElementById("stCustomSend");

let currentSelectionRange = null;

function handleSelectionChange() {
    if (!editorDoc) {
        hideSelectionToolbar();
        return;
    }
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        hideSelectionToolbar();
        return;
    }
    const range = selection.getRangeAt(0);
    if (!editorPage.contains(range.commonAncestorContainer)) {
        hideSelectionToolbar();
        return;
    }

    const text = selection.toString().trim();
    if (!text) {
        hideSelectionToolbar();
        return;
    }

    currentSelectionRange = range.cloneRange();

    const rect = range.getBoundingClientRect();
    if (editorSelectionToolbar) {
        editorSelectionToolbar.style.left = `${rect.left + rect.width / 2}px`;
        editorSelectionToolbar.style.top = `${rect.top}px`;
        editorSelectionToolbar.hidden = false;
    }
}

function hideSelectionToolbar() {
    if (editorSelectionToolbar) editorSelectionToolbar.hidden = true;
}

document.addEventListener("selectionchange", handleSelectionChange);
window.addEventListener("scroll", hideSelectionToolbar, true);
window.addEventListener("resize", hideSelectionToolbar);

document.addEventListener("mousedown", (e) => {
    if (editorSelectionToolbar && !editorSelectionToolbar.contains(e.target) && !editorPage.contains(e.target)) {
        // Delay slightly to allow button click events to fire before hiding
        setTimeout(() => {
            const activeSel = window.getSelection();
            if (!activeSel || activeSel.isCollapsed) {
                hideSelectionToolbar();
            }
        }, 100);
    }
});

if (editorSelectionToolbar) {
    editorSelectionToolbar.querySelectorAll(".st-btn").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const action = btn.dataset.action;
            let prompt = "";
            if (action === "improve") prompt = "Improve the writing quality, tone, and grammar of this text.";
            else if (action === "shorten") prompt = "Make this text significantly shorter and more concise.";
            else if (action === "expand") prompt = "Expand this text, adding relevant details and professional language.";
            else if (action === "table") prompt = "Convert this plain text or lists into a well-structured markdown table.";

            await runSelectionEdit(prompt);
        });
    });

    if (stCustomSend) {
        stCustomSend.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const prompt = stCustomPrompt.value.trim();
            if (!prompt) return;
            await runSelectionEdit(prompt);
        });
    }

    if (stCustomPrompt) {
        stCustomPrompt.addEventListener("keydown", async (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                e.stopPropagation();
                const prompt = stCustomPrompt.value.trim();
                if (!prompt) return;
                await runSelectionEdit(prompt);
            }
        });
        stCustomPrompt.addEventListener("mousedown", (e) => {
            // Prevent selection toolbar from closing when clicking inside its input box
            e.stopPropagation();
        });
    }
}

async function runSelectionEdit(prompt) {
    if (!editorDoc || !currentSelectionRange) return;
    const selectedText = currentSelectionRange.toString().trim();
    if (!selectedText) return;

    const inputs = editorSelectionToolbar.querySelectorAll("button, input");
    inputs.forEach(el => el.disabled = true);
    stCustomPrompt.placeholder = "Processing...";
    stCustomPrompt.value = "";

    try {
        const res = await fetch(`/documents/${encodeURIComponent(editorDoc.source)}/edit-chat`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({
                query: prompt,
                current_markdown: selectedText
            })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || "AI edit failed.");

        replaceSelectionWithContent(data.after || data.markdown || "");
    } catch (err) {
        alert(err.message);
    } finally {
        inputs.forEach(el => el.disabled = false);
        stCustomPrompt.placeholder = "Ask AI to edit selection...";
        hideSelectionToolbar();
    }
}

function replaceSelectionWithContent(newContent) {
    if (!currentSelectionRange) return;
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(currentSelectionRange);

    const html = renderMarkdown(newContent);
    document.execCommand("insertHTML", false, html);

    hideSelectionToolbar();
    triggerAutoSave();
}

/* ── Slash Command Menu ───────────────────────────────────────── */
const editorSlashMenu = document.getElementById("editorSlashMenu");
let slashMenuTriggered = false;
let slashFilterText = "";

function showSlashMenu(coords) {
    if (!editorSlashMenu) return;
    editorSlashMenu.style.left = `${coords.left}px`;
    editorSlashMenu.style.top = `${coords.top}px`;
    editorSlashMenu.hidden = false;
    slashMenuTriggered = true;
    slashFilterText = "";
    filterSlashMenu();
}

function hideSlashMenu() {
    if (editorSlashMenu) {
        editorSlashMenu.hidden = true;
        slashMenuTriggered = false;
        slashFilterText = "";
    }
}

function filterSlashMenu() {
    if (!editorSlashMenu) return;
    const items = Array.from(editorSlashMenu.querySelectorAll(".sm-item"));
    let visibleCount = 0;

    items.forEach((item) => {
        const name = item.querySelector(".sm-item-name").textContent.toLowerCase();
        const desc = item.querySelector(".sm-item-desc").textContent.toLowerCase();

        const matches = name.includes(slashFilterText.toLowerCase()) || desc.includes(slashFilterText.toLowerCase());
        item.style.display = matches ? "flex" : "none";

        if (matches) {
            item.classList.toggle("is-active", visibleCount === 0);
            visibleCount++;
        } else {
            item.classList.remove("is-active");
        }
    });

    if (visibleCount === 0) {
        editorSlashMenu.hidden = true;
    } else {
        editorSlashMenu.hidden = false;
    }
}

function getSelectionCoords() {
    const sel = window.getSelection();
    if (sel.rangeCount === 0) return { left: 0, top: 0 };
    const range = sel.getRangeAt(0).cloneRange();
    range.collapse(true);
    const rect = range.getBoundingClientRect();
    return { left: rect.left, top: rect.bottom }; // Position below cursor
}

function executeSlashCommand(command) {
    const sel = window.getSelection();
    if (sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);

    const node = range.startContainer;
    const offset = range.startOffset;
    if (node.nodeType === Node.TEXT_NODE) {
        const text = node.nodeValue;
        const slashIdx = text.lastIndexOf("/", offset);
        if (slashIdx !== -1) {
            range.setStart(node, slashIdx);
            range.setEnd(node, offset);
            range.deleteContents();
        }
    }

    let html = "";
    if (command === "h1") html = "<h1>Heading 1</h1>";
    else if (command === "h2") html = "<h2>Heading 2</h2>";
    else if (command === "h3") html = "<h3>Heading 3</h3>";
    else if (command === "bullet") html = "<ul><li>List item</li></ul>";
    else if (command === "number") html = "<ol><li>List item</li></ol>";
    else if (command === "table") {
        html = "<table><thead><tr><th>Header 1</th><th>Header 2</th></tr></thead><tbody><tr><td>Cell 1</td><td>Cell 2</td></tr></tbody></table>";
    } else if (command === "kpi") {
        html = `<div class="kpi-card" contenteditable="false"><span class="kpi-value" contenteditable="true">12.5K</span><span class="kpi-label" contenteditable="true">KPI Metric</span></div>`;
    } else if (command === "break") {
        html = `<div class="page-break" style="page-break-after: always;" contenteditable="false"></div>`;
    }

    document.execCommand("insertHTML", false, html);
    hideSlashMenu();
    triggerAutoSave();
}

editorPage.addEventListener("keydown", (e) => {
    if (slashMenuTriggered) {
        const activeItem = editorSlashMenu.querySelector(".sm-item.is-active");

        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            const items = Array.from(editorSlashMenu.querySelectorAll(".sm-item")).filter(i => i.style.display !== "none");
            if (items.length === 0) return;

            let idx = items.indexOf(activeItem);
            if (e.key === "ArrowDown") idx = (idx + 1) % items.length;
            else idx = (idx - 1 + items.length) % items.length;

            items.forEach((item, i) => item.classList.toggle("is-active", i === idx));
            return;
        }

        if (e.key === "Enter") {
            e.preventDefault();
            if (activeItem) {
                executeSlashCommand(activeItem.dataset.command);
            }
            return;
        }

        if (e.key === "Escape") {
            e.preventDefault();
            hideSlashMenu();
            return;
        }

        if (e.key === " ") {
            hideSlashMenu();
            return;
        }
    }
});

editorPage.addEventListener("keyup", (e) => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) {
        hideSlashMenu();
        return;
    }

    const range = selection.getRangeAt(0);
    const node = range.startContainer;
    const offset = range.startOffset;

    if (node.nodeType !== Node.TEXT_NODE) {
        hideSlashMenu();
        return;
    }

    const text = node.nodeValue;
    const slashIdx = text.lastIndexOf("/", offset);

    if (slashIdx === -1) {
        hideSlashMenu();
        return;
    }

    const beforeSlash = text.substring(0, slashIdx);
    const hasSpaceBefore = beforeSlash.length === 0 || /\s$/.test(beforeSlash);

    if (!hasSpaceBefore) {
        hideSlashMenu();
        return;
    }

    const filter = text.substring(slashIdx + 1, offset);

    if (e.key === "/" && !slashMenuTriggered) {
        const coords = getSelectionCoords();
        showSlashMenu(coords);
    } else if (slashMenuTriggered) {
        if (offset < slashIdx) {
            hideSlashMenu();
        } else {
            slashFilterText = filter;
            filterSlashMenu();
        }
    }
});

document.addEventListener("mousedown", (e) => {
    if (editorSlashMenu && !editorSlashMenu.contains(e.target)) {
        hideSlashMenu();
    }
});

if (editorSlashMenu) {
    editorSlashMenu.querySelectorAll(".sm-item").forEach((item) => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            executeSlashCommand(item.dataset.command);
        });
    });
}

/* ── Table formatting tools ──────────────────────────────────── */
const editorTableTools = document.getElementById("editorTableTools");
const tableToolsDivider = document.querySelector(".table-tools-divider");

function updateTableToolsVisibility() {
    if (!editorTableTools) return;
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        let container = range.commonAncestorContainer;
        if (container.nodeType === Node.TEXT_NODE) {
            container = container.parentNode;
        }
        const cell = container.closest ? container.closest("td, th") : null;
        if (cell && editorPage.contains(cell)) {
            editorTableTools.hidden = false;
            if (tableToolsDivider) tableToolsDivider.hidden = false;
            return;
        }
    }
    editorTableTools.hidden = true;
    if (tableToolsDivider) tableToolsDivider.hidden = true;
}

document.addEventListener("selectionchange", updateTableToolsVisibility);

if (editorTableTools) {
    document.getElementById("tableAddRowAbove").addEventListener("click", (e) => { e.preventDefault(); addTableRow(true); });
    document.getElementById("tableAddRowBelow").addEventListener("click", (e) => { e.preventDefault(); addTableRow(false); });
    document.getElementById("tableDeleteRow").addEventListener("click", (e) => { e.preventDefault(); deleteTableRow(); });
    document.getElementById("tableAddColLeft").addEventListener("click", (e) => { e.preventDefault(); addTableColumn(true); });
    document.getElementById("tableAddColRight").addEventListener("click", (e) => { e.preventDefault(); addTableColumn(false); });
    document.getElementById("tableDeleteCol").addEventListener("click", (e) => { e.preventDefault(); deleteTableColumn(); });
}

function getActiveTableCell() {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;
    const range = selection.getRangeAt(0);
    let container = range.commonAncestorContainer;
    if (container.nodeType === Node.TEXT_NODE) container = container.parentNode;
    const cell = container.closest ? container.closest("td, th") : null;
    return (cell && editorPage.contains(cell)) ? cell : null;
}

function addTableRow(above = false) {
    const cell = getActiveTableCell();
    if (!cell) return;
    const row = cell.closest("tr");
    const colCount = row.cells.length;

    const newRow = document.createElement("tr");
    for (let i = 0; i < colCount; i++) {
        const newCell = document.createElement(cell.tagName.toLowerCase() === "th" ? "th" : "td");
        newCell.innerHTML = "<br>";
        newRow.appendChild(newCell);
    }

    if (above) {
        row.parentNode.insertBefore(newRow, row);
    } else {
        row.parentNode.insertBefore(newRow, row.nextSibling);
    }
    triggerAutoSave();
}

function deleteTableRow() {
    const cell = getActiveTableCell();
    if (!cell) return;
    const row = cell.closest("tr");
    const table = row.closest("table");

    if (table.rows.length <= 1) {
        table.remove();
    } else {
        row.remove();
    }
    triggerAutoSave();
}

function addTableColumn(left = false) {
    const cell = getActiveTableCell();
    if (!cell) return;
    const cellIdx = cell.cellIndex;
    const table = cell.closest("table");

    Array.from(table.rows).forEach((row) => {
        const targetCell = row.cells[cellIdx];
        if (targetCell) {
            const newCell = document.createElement(targetCell.tagName.toLowerCase());
            newCell.innerHTML = "<br>";
            if (left) {
                row.insertBefore(newCell, targetCell);
            } else {
                row.insertBefore(newCell, targetCell.nextSibling);
            }
        }
    });
    triggerAutoSave();
}

function deleteTableColumn() {
    const cell = getActiveTableCell();
    if (!cell) return;
    const cellIdx = cell.cellIndex;
    const table = cell.closest("table");
    const colCount = cell.closest("tr").cells.length;

    if (colCount <= 1) {
        table.remove();
    } else {
        Array.from(table.rows).forEach((row) => {
            if (row.cells[cellIdx]) {
                row.cells[cellIdx].remove();
            }
        });
    }
    triggerAutoSave();
}

/* ── Image resize toolbar ────────────────────────────────────── */
const editorImageResizeMenu = document.createElement("div");
editorImageResizeMenu.className = "image-resize-menu";
editorImageResizeMenu.hidden = true;
editorImageResizeMenu.innerHTML = `
    <button type="button" data-size="25%">25%</button>
    <button type="button" data-size="50%">50%</button>
    <button type="button" data-size="75%">75%</button>
    <button type="button" data-size="100%">100%</button>
    <div class="st-divider"></div>
    <button type="button" id="imgDeleteBtn" class="danger"><span class="material-symbols-outlined">delete</span></button>
`;
document.body.appendChild(editorImageResizeMenu);

let currentResizingImage = null;

editorPage.addEventListener("click", (e) => {
    // 1. Image Clicks for resizing
    if (e.target.tagName === "IMG" && !e.target.closest(".report-chart-container")) {
        currentResizingImage = e.target;
        const rect = e.target.getBoundingClientRect();
        editorImageResizeMenu.style.left = `${rect.left + rect.width / 2}px`;
        editorImageResizeMenu.style.top = `${rect.top}px`;
        editorImageResizeMenu.hidden = false;
        e.stopPropagation();
    } else {
        if (!e.target.closest(".image-resize-menu")) {
            editorImageResizeMenu.hidden = true;
            currentResizingImage = null;
        }
    }

    // 2. Chart Clicks for re-editing
    const chartContainer = e.target.closest(".report-chart-container");
    if (chartContainer) {
        e.preventDefault();
        e.stopPropagation();
        const specStr = chartContainer.dataset.spec;
        if (specStr) {
            try {
                const spec = JSON.parse(specStr);
                if (chartTypeEl) chartTypeEl.value = spec.type || "bar";
                if (chartTitleEl) chartTitleEl.value = spec.title || "";
                if (chartDataEl) {
                    const lines = (spec.labels || []).map((l, i) => `${l}, ${spec.values[i] || 0}`);
                    chartDataEl.value = lines.join("\n");
                }
                renderChartPreview();

                currentEditingChartContainer = chartContainer;
                if (chartInsertBtn) {
                    chartInsertBtn.innerHTML = `<span class="material-symbols-outlined">edit</span> Update Chart`;
                }

                if (window.matchMedia("(max-width: 1180px)").matches) {
                    editorView.classList.add("report-tools-open");
                }
            } catch (err) {
                console.error("Failed to parse chart spec:", err);
            }
        }
    } else {
        if (!e.target.closest("#reportTools") && currentEditingChartContainer) {
            currentEditingChartContainer = null;
            if (chartInsertBtn) {
                chartInsertBtn.innerHTML = `<span class="material-symbols-outlined">insert_chart</span> Insert chart`;
            }
        }
    }
});

editorImageResizeMenu.querySelectorAll("button[data-size]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
        e.preventDefault();
        if (currentResizingImage) {
            const size = btn.dataset.size;
            currentResizingImage.style.width = size;
            currentResizingImage.style.height = "auto";
            currentResizingImage.style.maxWidth = "100%";
            triggerAutoSave();
        }
        editorImageResizeMenu.hidden = true;
        currentResizingImage = null;
    });
});

const imgDeleteBtn = document.getElementById("imgDeleteBtn");
if (imgDeleteBtn) {
    imgDeleteBtn.addEventListener("click", (e) => {
        e.preventDefault();
        if (currentResizingImage) {
            currentResizingImage.remove();
            triggerAutoSave();
        }
        editorImageResizeMenu.hidden = true;
        currentResizingImage = null;
    });
}

/* ── Outline & TOC panel ─────────────────────────────────────── */
function rebuildOutline() {
    if (!outlineList || !editorPage) return;
    outlineList.innerHTML = "";

    const headings = Array.from(editorPage.querySelectorAll("h1, h2, h3"));
    if (headings.length === 0) {
        outlineList.innerHTML = `<p style="font-size: 11px; color: var(--faint); padding: 0 8px; line-height: 1.4;">No headings yet. Use Heading levels to structure your document.</p>`;
        return;
    }

    headings.forEach((heading, idx) => {
        if (!heading.id) {
            heading.id = `heading-${idx}-${Date.now().toString(36)}`;
        }

        const item = document.createElement("div");
        const level = parseInt(heading.tagName[1]);
        item.className = `outline-item level-${level}`;
        item.textContent = heading.textContent.trim() || `Heading ${level}`;
        item.title = heading.textContent.trim() || `Heading ${level}`;
        item.addEventListener("click", () => {
            heading.scrollIntoView({ behavior: "smooth", block: "start" });
            heading.style.transition = "background-color 0.3s";
            heading.style.backgroundColor = "var(--accent-soft)";
            setTimeout(() => heading.style.backgroundColor = "", 1000);
        });
        outlineList.appendChild(item);
    });
}

function insertTOC() {
    const headings = Array.from(editorPage.querySelectorAll("h1, h2, h3"));
    if (headings.length === 0) {
        alert("Please add some headings to generate a Table of Contents.");
        return;
    }

    let html = `<div class="table-of-contents" contenteditable="false">`;
    html += `<h4>Table of Contents</h4><ul>`;
    headings.forEach((h) => {
        if (!h.id) {
            h.id = `heading-${Math.random().toString(36).substr(2, 9)}`;
        }
        const level = parseInt(h.tagName[1]);
        const indent = level > 1 ? ` style="margin-left: ${(level - 1) * 16}px;"` : "";
        html += `<li${indent}><a href="#${h.id}" class="toc-link">${escapeHtml(h.textContent.trim())}</a></li>`;
    });
    html += `</ul></div><p><br></p>`;

    document.execCommand("insertHTML", false, html);
    rebuildOutline();
    triggerAutoSave();
}

if (editorOutlineToggleBtn) {
    editorOutlineToggleBtn.addEventListener("click", (e) => {
        e.preventDefault();
        if (editorOutlinePanel) {
            editorOutlinePanel.hidden = !editorOutlinePanel.hidden;
            rebuildOutline();
        }
    });
}

if (insertTOCBtn) {
    insertTOCBtn.addEventListener("click", (e) => {
        e.preventDefault();
        insertTOC();
    });
}

// Intercept smooth scroll for TOC links & open citation panel for cite links
editorPage.addEventListener("click", (e) => {
    const link = e.target.closest(".toc-link");
    if (link) {
        e.preventDefault();
        const targetId = link.getAttribute("href").slice(1);
        const target = editorPage.querySelector(`#${targetId}`);
        if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    const cite = e.target.closest(".citation-link");
    if (cite) {
        e.preventDefault();
        const filename = cite.dataset.filename;
        const snippet = cite.dataset.snippet;
        const srcObj = {
            source: filename,
            snippet: snippet,
            content: snippet,
            parent_content: snippet
        };
        openCitationPanel(srcObj, [srcObj]);
    }
});

// Register change scanner
editorPage.addEventListener("input", rebuildOutline);

chatForm.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });

window.addEventListener("load", () => {
    bindSuggestions();
    scrollToBottom();
    updateProgress(0, "Ready", "Idle");
    logEvent("idle", "Checking system status…");

    // Gate input until models are loaded and the index is built.
    pollStatus();
    statusPollTimer = setInterval(pollStatus, 3000);

    loadLibrary();
});
