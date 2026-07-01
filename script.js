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
const editorPage        = document.getElementById("editorPage");
const editorToolbar     = document.getElementById("editorToolbar");
const editorSaveStatus  = document.getElementById("editorSaveStatus");
const editorChatForm    = document.getElementById("editorChatForm");
const editorUserInput   = document.getElementById("editorUserInput");
const editorSendButton  = document.getElementById("editorSendButton");
const editorDownloadDocx = document.getElementById("editorDownloadDocx");
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
    querySource = "chat";
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

function renderAuthState() {
    if (!authButtonLabel) return;
    authButtonLabel.textContent = isAuthenticated() ? (authUser || "Sign out") : "Sign in";
    if (authButton) authButton.title = isAuthenticated()
        ? `Signed in as ${authUser} — click to sign out`
        : "Sign in";
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

        try {
            const res = await fetch(AUTH_LOGIN_URL, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ username, password }),
            });
            const data = await res.json();
            if (!res.ok) {
                authError.textContent = data.error || "Sign-in failed.";
                authError.hidden = false;
                return;
            }
            setAuth(data.token, data.username || username);
            closeAuthModal();
            logEvent("auth", `Signed in as ${data.username || username}.`);
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
    }
}

async function fetchWorkspaceDocs() {
    const res  = await fetch(`${ADMIN_DOCS_URL}?channel=workspace`, { headers: authHeaders() });
    if (!res.ok) throw new Error("fetch failed");
    const data = await res.json();
    return data.documents || [];
}

function renderDocRows(container, docs, activeSource, onSelect) {
    container.innerHTML = "";
    docs.forEach((doc) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "library-row";
        row.dataset.source = doc.source;
        if (activeSource === doc.source) row.classList.add("is-active");
        row.innerHTML = `
            <span class="material-symbols-outlined">description</span>
            <span class="library-row-text">
                <span class="library-row-name">${escapeHtml(doc.source)}</span>
                <span class="library-row-meta">${doc.chunks} chunk</span>
            </span>`;
        row.addEventListener("click", () => onSelect(doc.source));
        container.appendChild(row);
    });
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
                    <p>No documents yet. Upload a PDF/DOCX to get started.</p>
                </div>`;
            return;
        }
        renderDocRows(editorFileList, docs, editorDoc ? editorDoc.source : null, selectEditorDoc);
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

    citeState = { source, totalPages: null, currentPage: 0 };

    const data = await fetchCiteDoc(source, snippets, focusSnippet);
    loading.style.display = "none";
    if (data.error || !Array.isArray(data.pages) || data.pages.length === 0) {
        const msg = data.error || "Could not load citation view.";
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
let querySource = "chat";

// Auto resize for editor composer
function autoResizeEditorInput() {
    editorUserInput.style.height = "auto";
    editorUserInput.style.height = `${editorUserInput.scrollHeight}px`;
}
editorUserInput.addEventListener("input", autoResizeEditorInput);

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
                if (src.includes("/")) {
                    const parts = src.split("/");
                    src = parts[parts.length - 1];
                }
                text += `![${alt}](${src})`;
            } else {
                text += inner;
            }
        }
    }
    return text;
}

// Auto-save debounced handler
let autoSaveTimer = null;

function triggerAutoSave() {
    if (!editorDoc) return;
    setEditorSaveStatus("dirty");
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(async () => {
        await runAutoSave();
    }, 1200);
}

async function runAutoSave() {
    if (!editorDoc) return;
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
            const errData = await res.json();
            throw new Error(errData.error || "Could not save.");
        }
        const data = await res.json();
        editorDoc.markdown = currentMd;
        setEditorSaveStatus("saved");
    } catch (err) {
        console.error("Auto-save failed:", err);
        setEditorSaveStatus("error", err.message);
    }
}

function setEditorSaveStatus(state, errMsg = "") {
    editorSaveStatus.className = `editor-save-pill ${state}`;
    const textEl = editorSaveStatus.querySelector(".status-text");
    editorSaveStatus.hidden = false;
    if (state === "saved") {
        textEl.textContent = "Saved";
        setTimeout(() => {
            if (editorSaveStatus.classList.contains("saved")) {
                editorSaveStatus.hidden = true;
            }
        }, 3000);
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
        
        await runAutoSave();
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
async function sendEditorMessage() {
    const message = editorUserInput.value.trim();
    if (!message || editorSendButton.disabled) return;
    if (!editorDoc) return;
    
    editorUserInput.value = "";
    autoResizeEditorInput();
    
    const currentMd = htmlToMarkdown(editorPage);
    
    editorUserInput.disabled = true;
    editorSendButton.disabled = true;
    
    const originalContent = editorPage.innerHTML;
    editorPage.innerHTML = `
        <div class="editor-loading-overlay" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; gap: 16px; color: var(--muted);">
            <span class="material-symbols-outlined spin" style="font-size: 48px; animation: spin 2s linear infinite;">sync</span>
            <p style="font-size: 14px; font-weight: 500;">AI is computing changes, please wait…</p>
        </div>
    `;
    
    try {
        const res = await fetch(`/documents/${encodeURIComponent(editorDoc.source)}/edit-chat`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({
                query: message,
                current_markdown: currentMd
            })
        });
        
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || "Edit request failed.");
        }
        
        const data = await res.json();
        showWholeDocumentDiff(data.before, data.after);
        
    } catch (err) {
        alert(err.message);
        editorPage.innerHTML = originalContent;
    } finally {
        editorUserInput.disabled = false;
        editorSendButton.disabled = false;
        editorUserInput.focus();
    }
}


editorChatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendEditorMessage();
});

async function selectEditorDoc(source) {
    editorDoc = null;
    editorPage.hidden = true;
    editorDownloadDocx.disabled = true;
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
        
        // Enable download button
        editorDownloadDocx.disabled = false;
        editorDownloadDocx.onclick = () => {
            window.open(`/workspace/docx/${encodeURIComponent(source)}`, "_blank");
        };

        // Hide save status initially
        setEditorSaveStatus("saved");
    } catch (err) {
        console.error("selectEditorDoc error:", err);
        editorPage.innerHTML = `<p style="color: var(--danger); padding: 20px;">Could not load document.</p>`;
        editorPage.hidden = false;
    }
}

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
