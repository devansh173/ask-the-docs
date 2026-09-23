/* Ask the Docs - chat UI.
 *
 * No framework, no build step: this file plus a stylesheet, served as static
 * assets by the same FastAPI process that serves the API.
 *
 * Two things worth knowing:
 *
 * 1. An API key typed in the sidebar is kept in sessionStorage (this tab only)
 *    and sent with each request. The server uses it for that one call and drops
 *    it; it is never persisted server-side.
 * 2. Each embedding model has its own Qdrant collection, because vector
 *    dimensionality is fixed at collection creation. The UI therefore shows how
 *    many chunks each model has indexed and disables the ones with none, rather
 *    than letting you pick a model that would silently return nothing.
 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  providers: [],
  embeddings: [],
  rerankers: [],
  provider: null,
  model: null,
  embedding: null,
  reranker: null,
  corpus: "docs",
  config: "agentic",
  workspace: "default",
  documents: [],
  uploadChunks: 0,
  busy: false,
};

const KEY_STORE = "askthedocs.keys";

/* ------------------------------------------------------------------ utils */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const fmt = (n) => Number(n).toLocaleString();

function store(key, value) {
  try {
    if (value === undefined) return sessionStorage.getItem(key);
    sessionStorage.setItem(key, value);
  } catch {
    /* private mode: settings simply do not persist */
  }
}

function loadKeys() {
  try {
    return JSON.parse(sessionStorage.getItem(KEY_STORE) || "{}");
  } catch {
    return {};
  }
}

function saveKey(provider, key) {
  try {
    const keys = loadKeys();
    if (key) keys[provider] = key;
    else delete keys[provider];
    sessionStorage.setItem(KEY_STORE, JSON.stringify(keys));
  } catch {
    /* ignore */
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* Minimal markdown renderer.
 *
 * Everything is escaped first, so model output can never inject markup; the
 * patterns below then reintroduce a fixed set of safe tags. Citations like [2]
 * become chips wired to the source list.
 */
function renderMarkdown(text) {
  const blocks = [];
  let src = escapeHtml(text).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(
      `<pre><code class="lang-${escapeHtml(lang)}">${code.replace(/\n$/, "")}</code></pre>`
    );
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });

  src = src
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![\w*])\*([^*\n]+)\*(?![\w*])/g, "<em>$1</em>")
    // Models cite as "[1]" or grouped as "[2, 3]"; render each number as its
    // own chip so every one is clickable.
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) =>
      group
        .split(",")
        .map((n) => n.trim())
        .filter((n) => /^\d+$/.test(n))
        .map((n) => `<span class="cite" data-n="${n}">${n}</span>`)
        .join("")
    );

  const html = src
    .split(/\n{2,}/)
    .map((para) => {
      const trimmed = para.trim();
      if (!trimmed) return "";
      if (/^\u0000BLOCK\d+\u0000$/.test(trimmed)) return trimmed;

      const lines = trimmed.split("\n");
      if (lines.every((l) => /^\s*[-*]\s+/.test(l)))
        return `<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*]\s+/, "")}</li>`).join("")}</ul>`;
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l)))
        return `<ol>${lines.map((l) => `<li>${l.replace(/^\s*\d+[.)]\s+/, "")}</li>`).join("")}</ol>`;
      if (/^#{1,4}\s/.test(trimmed))
        return `<p><strong>${trimmed.replace(/^#{1,4}\s/, "")}</strong></p>`;
      return `<p>${lines.join("<br>")}</p>`;
    })
    .join("");

  return html.replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

const NODE_LABELS = {
  analyze_query: "Analysing question",
  retrieve: "Searching the index",
  rerank: "Reranking passages",
  truncate: "Selecting passages",
  grade_documents: "Grading relevance",
  rewrite_query: "Rewriting the query",
  generate: "Writing the answer",
  check_groundedness: "Checking groundedness",
  mark_regenerate: "Rewriting — claims unsupported",
  give_up: "Not enough in the index",
};

function stepMeta(ev) {
  const parts = [];
  if (ev.mode) parts.push(ev.mode);
  if (ev.returned != null) parts.push(`${ev.returned} hits`);
  if (ev.kept != null) parts.push(`kept ${ev.kept}`);
  if (ev.score != null) parts.push(`score ${ev.score}`);
  if (ev.sufficient != null) parts.push(ev.sufficient ? "sufficient" : "insufficient");
  if (ev.grounded != null) parts.push(ev.grounded ? "grounded" : "unsupported claims");
  if (ev.intent) parts.push(ev.intent);
  if (ev.cited != null) parts.push(`${ev.cited} citations`);
  if (ev.attempt > 1) parts.push(`attempt ${ev.attempt}`);
  if (ev.skipped) parts.push(String(ev.skipped));
  return parts.join(" · ");
}

/* ----------------------------------------------------------------- status */

async function loadHealth() {
  const dl = $("#status");
  try {
    const h = await api("/api/health");
    const rows = [
      ["Status", h.status === "ok" ? "ready" : h.status, h.status === "ok" ? "ok" : "warn"],
      ["Chunks", fmt(h.indexed_chunks), ""],
      ["Pages", fmt(h.corpus_pages), ""],
      ["Dense", h.embedding_model.split("/").pop(), "mono"],
      ["Sparse", h.sparse_model.split("/").pop(), "mono"],
      ["Reranker", h.reranker_model.split("/").pop(), "mono"],
      ["Qdrant", h.qdrant.replace(/^https?:\/\//, "").split("/")[0], "mono"],
      ["Tracing", h.langfuse ? "langfuse" : "off", h.langfuse ? "ok" : ""],
    ];
    dl.innerHTML = rows
      .map(([k, v, cls]) =>
        `<dt>${escapeHtml(k)}</dt><dd class="${cls}" title="${escapeHtml(v)}">${escapeHtml(v)}</dd>`)
      .join("");
  } catch {
    dl.innerHTML = '<dt>Status</dt><dd class="warn">API unreachable</dd>';
  }
}

/* -------------------------------------------------------------- providers */

async function loadProviders() {
  state.providers = await api("/api/providers");
  const sel = $("#provider");
  sel.innerHTML = state.providers
    .map((p) => `<option value="${p.id}">${escapeHtml(p.label)}</option>`)
    .join("");

  const saved = store("askthedocs.provider");
  state.provider = state.providers.find((p) => p.id === saved)?.id || state.providers[0]?.id;
  sel.value = state.provider;
  syncProvider();
}

const currentProvider = () => state.providers.find((p) => p.id === state.provider);

function syncProvider() {
  const p = currentProvider();
  if (!p) return;

  $("#model").innerHTML = p.models
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`)
    .join("");
  state.model = p.default_model;
  $("#model").value = state.model;

  $("#key-field").style.display = p.needs_key ? "" : "none";
  $("#api-key").value = loadKeys()[p.id] || "";

  $("#key-hint").textContent = !p.needs_key
    ? p.notes || "No API key needed."
    : p.key_in_env
    ? "A key is set on the server — leave this blank to use it."
    : "Kept in this browser tab only. Sent per request, never stored on the server.";

  store("askthedocs.provider", p.id);
}

/* --------------------------------------------------------------- retrieval */

async function loadModels() {
  const catalog = await api("/api/models");
  state.embeddings = catalog.embeddings;
  state.rerankers = catalog.rerankers;

  const saved = store("askthedocs.embedding");
  const usable = state.embeddings.filter((e) => e.available && e.indexed_chunks > 0);
  state.embedding =
    state.embeddings.find((e) => e.slug === saved && e.available)?.slug ||
    usable[0]?.slug ||
    catalog.default_embedding;

  $("#embedding").innerHTML = state.embeddings
    .map((e) => {
      const indexed = e.indexed_chunks > 0 ? `${fmt(e.indexed_chunks)} chunks` : "not indexed";
      const disabled = !e.available ? " disabled" : "";
      return `<option value="${e.slug}"${disabled}>${escapeHtml(e.label)} · ${e.dim}d · ${indexed}</option>`;
    })
    .join("");
  $("#embedding").value = state.embedding;
  syncEmbedding();

  const savedRer = store("askthedocs.reranker");
  state.reranker =
    state.rerankers.find((r) => r.slug === savedRer && r.available)?.slug ||
    catalog.default_reranker;

  $("#reranker").innerHTML = state.rerankers
    .map((r) => {
      const size = r.size_mb ? ` · ${r.size_mb} MB` : "";
      return `<option value="${r.slug}"${r.available ? "" : " disabled"}>${escapeHtml(r.label)}${size}</option>`;
    })
    .join("");
  $("#reranker").value = state.reranker;
  syncReranker();
}

function currentEmbedding() {
  return state.embeddings.find((e) => e.slug === state.embedding);
}

function syncEmbedding() {
  const e = currentEmbedding();
  if (!e) return;
  const bits = [`${e.dim}-dim`, `${e.size_mb} MB`, e.backend];
  let hint = `<strong>${escapeHtml(bits.join(" · "))}</strong><br>${escapeHtml(e.notes || "")}`;
  if (!e.available) {
    hint += `<br>Unavailable: ${escapeHtml(e.unavailable_reason)}`;
  } else if (!e.indexed_chunks) {
    hint += `<br>Nothing indexed yet. Build it with <code>askthedocs index --recreate --embedding ${escapeHtml(e.slug)}</code>`;
  }
  $("#embedding-hint").innerHTML = hint;
  store("askthedocs.embedding", e.slug);
  updateComposerNote();
}

function syncReranker() {
  const r = state.rerankers.find((x) => x.slug === state.reranker);
  if (!r) return;
  $("#reranker-hint").textContent = r.notes || "";
  store("askthedocs.reranker", r.slug);
}

/* ----------------------------------------------------------------- uploads */

function renderDocuments() {
  const list = $("#doclist");
  list.innerHTML = state.documents
    .map(
      (d) => `<li>
        <span class="name" title="${escapeHtml(d.filename)}">${escapeHtml(d.title || d.filename)}</span>
        <span class="meta">${fmt(d.chunks)} ch${d.pages ? ` · ${d.pages}p` : ""}</span>
      </li>`
    )
    .join("");
  $("#clear-docs").hidden = state.documents.length === 0;
  updateComposerNote();
}

async function refreshWorkspace() {
  try {
    const info = await api(
      `/api/workspace?workspace=${encodeURIComponent(state.workspace)}&embedding=${encodeURIComponent(state.embedding)}`
    );
    state.documents = info.documents || [];
    state.uploadChunks = info.total_chunks || 0;
  } catch {
    state.documents = [];
    state.uploadChunks = 0;
  }
  renderDocuments();
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;

  const progress = $("#upload-progress");
  const fill = $("#upload-fill");
  const status = $("#upload-status");
  progress.hidden = false;
  progress.classList.remove("indeterminate");
  fill.style.width = "0%";

  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("workspace", state.workspace);
  form.append("embedding", state.embedding);

  try {
    // XHR rather than fetch: it reports upload progress, which matters because
    // a large PDF spends real time on the wire before indexing even starts.
    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const pct = Math.round((e.loaded / e.total) * 100);
        fill.style.width = `${pct}%`;
        status.textContent = pct < 100 ? `Uploading… ${pct}%` : "Embedding and indexing…";
        if (pct >= 100) progress.classList.add("indeterminate");
      };
      xhr.onload = () => {
        try {
          const body = JSON.parse(xhr.responseText);
          xhr.status >= 200 && xhr.status < 300
            ? resolve(body)
            : reject(new Error(body.detail || `Upload failed (${xhr.status})`));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(form);
    });

    state.documents = result.documents || [];
    state.uploadChunks = result.total_chunks || 0;
    renderDocuments();

    status.textContent = `Indexed ${fmt(result.chunks_indexed)} chunks in ${(result.elapsed_ms / 1000).toFixed(1)}s`;
    progress.classList.remove("indeterminate");
    fill.style.width = "100%";
    setTimeout(() => (progress.hidden = true), 2500);

    // Uploading implies wanting to ask about them.
    selectCorpus("uploads");
    await loadModels();
  } catch (err) {
    progress.classList.remove("indeterminate");
    progress.hidden = true;
    showBanner(err.message, "error");
  }
}

function showBanner(message, kind = "error") {
  $("#welcome")?.remove();
  const el = document.createElement("div");
  el.className = "turn";
  el.innerHTML = `<div class="${kind === "error" ? "error" : "flag"}">${escapeHtml(message)}</div>`;
  $("#thread").append(el);
  el.scrollIntoView({ block: "end" });
}

/* ------------------------------------------------------------------ corpus */

function selectCorpus(corpus) {
  state.corpus = corpus;
  $$(".seg[data-corpus]").forEach((b) =>
    b.classList.toggle("active", b.dataset.corpus === corpus)
  );
  $("#pane-docs").hidden = corpus !== "docs";
  $("#pane-uploads").hidden = corpus !== "uploads";
  updateComposerNote();
}

function updateComposerNote() {
  const note = $("#composer-note");
  if (!note) return;

  if (state.corpus === "uploads") {
    note.textContent = state.uploadChunks
      ? `Answering from your ${state.documents.length} document${state.documents.length === 1 ? "" : "s"} (${fmt(state.uploadChunks)} chunks).`
      : "Upload a document to ask about your own files.";
    return;
  }
  const e = currentEmbedding();
  note.textContent = e && !e.indexed_chunks
    ? `The ${e.label} index is empty — run askthedocs index --embedding ${e.slug}`
    : "";
}

/* -------------------------------------------------------------------- ask */

function newTurn(question) {
  $("#welcome")?.remove();
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="q">${escapeHtml(question)}</div>
    <div class="live"><span class="dot"></span><span class="live-text">Starting…</span></div>
    <div class="a"></div>
    <div class="extras"></div>`;
  $("#thread").append(turn);
  turn.scrollIntoView({ block: "start" });
  return turn;
}

function renderCitations(turn, citations) {
  if (!citations?.length) return;
  const html = citations
    .map((c, i) => {
      const label = `${escapeHtml(c.title)}${c.section ? " › " + escapeHtml(c.section) : ""}`;
      const score = Number(c.score).toFixed(3);
      // Uncited passages are still listed so [n] lines up with source n, but
      // dimmed: they are what the model read and chose not to use.
      const dim = c.cited === false ? " uncited" : "";
      const inner = `<span class="n${dim}">${i + 1}</span><span class="t">${label}</span><span class="sc">${score}</span>`;
      // Uploaded documents have no public URL, so they are not links.
      return c.source_url
        ? `<a class="source${dim}" id="src-${i + 1}" href="${escapeHtml(c.source_url)}" target="_blank" rel="noopener noreferrer">${inner}</a>`
        : `<div class="source${dim}" id="src-${i + 1}">${inner}</div>`;
    })
    .join("");
  turn.querySelector(".extras").insertAdjacentHTML(
    "beforeend",
    `<div class="sources"><h3>Sources</h3>${html}</div>`
  );
}

function renderTrace(turn, events, meta) {
  if (!events?.length) return;
  const steps = events
    .map(
      (ev) => `<div class="step">
        <span class="name">${escapeHtml(NODE_LABELS[ev.node] || ev.node)}</span>
        <span class="ms">${ev.ms} ms</span>
        <span class="meta">${escapeHtml(stepMeta(ev))}</span>
      </div>`
    )
    .join("");

  const total = events.reduce((sum, e) => sum + (e.ms || 0), 0);
  const summary = [
    `${events.length} steps`,
    `${Math.round(total)} ms`,
    meta.retrieval_mode,
    meta.attempts > 1 ? `${meta.attempts} retrieval attempts` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  turn.querySelector(".extras").insertAdjacentHTML(
    "beforeend",
    `<details class="trace"><summary>Pipeline — ${escapeHtml(summary)}</summary>
       <div class="steps">${steps}</div></details>`
  );
}

function wireCitations(turn) {
  turn.querySelectorAll(".cite").forEach((chip) => {
    chip.addEventListener("click", () => {
      const target = turn.querySelector(`#src-${chip.dataset.n}`);
      if (!target) return;
      target.scrollIntoView({ block: "nearest" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 900);
    });
  });
}

async function ask(question) {
  if (state.busy) return;
  state.busy = true;
  $("#send").disabled = true;

  const turn = newTurn(question);
  const live = turn.querySelector(".live");
  const liveText = turn.querySelector(".live-text");
  const answerEl = turn.querySelector(".a");

  const docSets = $$("#doc-sets input:checked").map((i) => i.value);

  let buffer = "";
  const collected = [];

  const finish = (payload) => {
    live.remove();
    answerEl.innerHTML = renderMarkdown(payload.streamed ? buffer : payload.answer || buffer);

    if (payload.grounded === false) {
      answerEl.insertAdjacentHTML("afterbegin",
        `<div class="flag">The groundedness check could not match every claim in this
         answer to a cited passage. Treat the unmatched parts with suspicion.</div>`);
    }
    if (payload.status === "insufficient_context") {
      answerEl.insertAdjacentHTML("afterbegin",
        `<div class="flag">The pipeline stopped rather than answering from weak context${
          payload.attempts > 1 ? ` (after ${payload.attempts} retrieval attempts)` : ""
        }.</div>`);
    }
    renderCitations(turn, payload.citations);
    renderTrace(turn, payload.events || collected, payload);
    wireCitations(turn);
  };

  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        provider: state.provider,
        model: state.model,
        api_key: $("#api-key").value.trim() || null,
        embedding: state.embedding,
        reranker: state.reranker,
        corpus: state.corpus,
        workspace: state.workspace,
        doc_sets: state.corpus === "docs" ? docSets : [],
        config: state.config,
      }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      const frames = pending.split("\n\n");
      pending = frames.pop();

      for (const frame of frames) {
        const evLine = frame.match(/^event: (.+)$/m);
        const dataLine = frame.match(/^data: (.+)$/m);
        if (!evLine || !dataLine) continue;

        let data;
        try {
          data = JSON.parse(dataLine[1]);
        } catch {
          continue;
        }

        switch (evLine[1]) {
          case "token":
            buffer += data.text;
            answerEl.innerHTML = renderMarkdown(buffer) + '<span class="cursor"></span>';
            $("#thread").scrollTop = $("#thread").scrollHeight;
            break;
          case "node":
            collected.push(data);
            liveText.textContent = NODE_LABELS[data.node] || data.node;
            break;
          case "reset":
            buffer = "";
            answerEl.innerHTML = "";
            liveText.textContent = NODE_LABELS.mark_regenerate;
            break;
          case "done":
            finish({ ...data, events: collected });
            break;
          case "error":
            throw new Error(data.detail);
        }
      }
    }
  } catch (err) {
    live.remove();
    answerEl.innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`;
  } finally {
    state.busy = false;
    $("#send").disabled = false;
    $("#question").focus();
  }
}

/* ------------------------------------------------------------------ setup */

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}

function toggleSidebar(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#scrim").hidden = !open;
}

async function init() {
  await Promise.all([
    loadProviders().catch(() => {}),
    loadModels().catch(() => {}),
    loadHealth(),
  ]);
  await refreshWorkspace();

  $("#provider").addEventListener("change", (e) => {
    state.provider = e.target.value;
    syncProvider();
  });
  $("#model").addEventListener("change", (e) => (state.model = e.target.value));
  $("#api-key").addEventListener("change", (e) => saveKey(state.provider, e.target.value.trim()));

  $("#embedding").addEventListener("change", async (e) => {
    state.embedding = e.target.value;
    syncEmbedding();
    // Uploads are indexed per embedding model, so the file list changes too.
    await refreshWorkspace();
  });
  $("#reranker").addEventListener("change", (e) => {
    state.reranker = e.target.value;
    syncReranker();
  });

  $$(".seg[data-corpus]").forEach((btn) =>
    btn.addEventListener("click", () => selectCorpus(btn.dataset.corpus))
  );

  $$(".seg[data-config]").forEach((btn) =>
    btn.addEventListener("click", () => {
      $$(".seg[data-config]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.config = btn.dataset.config;
      $("#config-hint").textContent =
        state.config === "agentic"
          ? "Hybrid retrieval → rerank → grade → rewrite & retry if weak → generate → groundedness check."
          : "Baseline: single-pass dense retrieval, top-k straight into the prompt. No rerank, no grading, no self-correction — what the eval table calls 'naive'.";
    })
  );

  // -- uploads --
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("over");
    })
  );
  dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

  $("#clear-docs").addEventListener("click", async () => {
    if (!confirm("Remove all uploaded documents from the index?")) return;
    await api(
      `/api/workspace?workspace=${encodeURIComponent(state.workspace)}&embedding=${encodeURIComponent(state.embedding)}`,
      { method: "DELETE" }
    ).catch((e) => showBanner(e.message));
    await refreshWorkspace();
    await loadModels();
  });

  // -- composer --
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#question").value.trim();
    if (!q) return;
    $("#question").value = "";
    autoGrow($("#question"));
    ask(q);
  });
  $("#question").addEventListener("input", (e) => autoGrow(e.target));
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#composer").requestSubmit();
    }
  });

  $("#examples")?.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") ask(e.target.textContent.trim());
  });

  $("#menu-btn").addEventListener("click", () => toggleSidebar(!$("#sidebar").classList.contains("open")));
  $("#scrim").addEventListener("click", () => toggleSidebar(false));
  $("#clear-btn").addEventListener("click", () => location.reload());

  $("#question").focus();
}

document.addEventListener("DOMContentLoaded", init);
