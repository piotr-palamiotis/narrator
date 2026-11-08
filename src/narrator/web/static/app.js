// Narrator web admin UI. Plain JS, no build step.
//
// Rules this file keeps:
//  - caption text (and anything else from the server) goes into the page
//    with textContent only, never innerHTML;
//  - no validation layer here: values go to the server as typed (numbers
//    parsed when they parse) and the Pydantic schema's errors come back
//    and are shown next to the fields;
//  - the form is generated from the schema (/api/schema), so every config
//    field is editable even before HINTS below knows about it.
"use strict";

// ------------------------------------------------------------------ helpers

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else if (key === "value") node.value = value; // a property: textarea content, live input value
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key in node && typeof value !== "string") node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function getPath(obj, path) {
  return path.reduce((o, k) => (o === undefined || o === null ? undefined : o[k]), obj);
}

function setPath(obj, path, value) {
  let o = obj;
  for (const k of path.slice(0, -1)) {
    if (o[k] === undefined || o[k] === null) o[k] = {};
    o = o[k];
  }
  o[path[path.length - 1]] = value;
}

const clone = (v) => JSON.parse(JSON.stringify(v));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

async function api(method, path, body) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (res.status === 401) {
    location.reload(); // session gone (server restarted): back to the login page
    throw new Error("not logged in");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  return { status: res.status, ok: res.ok, data };
}

let toastTimer = null;
function toast(message, isError = false) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.toggle("error", isError);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 5000 : 1800);
}

// ------------------------------------------------------------------ state

const S = {
  schema: null,
  config: null, // working copy (form)
  saved: null, // last saved / loaded
  choices: { models: [], fonts: [], languages: [], presets: [], vocabulary_lists: [] },
  devices: [],
  state: {},
  rev: null,
  shortcuts: {},
  errors: {}, // "asr.step_ms" -> message
  activeTab: null,
  ws: null,
  logLines: [],
  feed: [],
};

const DISPLAY_SECTIONS = ["canvas", "layout", "style"];

// ------------------------------------------------------------------ schema UI

const TABS = [
  { id: "audio", title: "Audio", sections: ["audio"] },
  { id: "asr", title: "ASR", sections: ["asr"] },
  { id: "display", title: "Display", sections: ["canvas", "layout", "style"] },
  { id: "ndi", title: "NDI", sections: ["ndi"] },
  { id: "transcript", title: "Transcript", sections: ["transcript"] },
  { id: "captions", title: "Captions", sections: ["captions"] },
  { id: "recordings", title: "Recordings", sections: ["recording"] },
  { id: "remote", title: "Web & remote", sections: ["web"] },
  { id: "status", title: "Status & log", sections: ["app"] },
];

const SECTION_TITLES = {
  app: "App", audio: "Audio input", asr: "Speech recognition", canvas: "Canvas", layout: "Layout",
  style: "Style", ndi: "NDI output", transcript: "Transcript file", captions: "Caption text policy",
  web: "Web UI & remote control", recording: "Recording settings", overrides: "Style overrides", background_band: "Background band",
  shadow: "Shadow", stroke: "Stroke (outline)", shortcuts: "Keyboard shortcuts", osc: "OSC input",
};

// Fields only one decoder reads are shown only while it is selected (`when`:
// every listed path must hold one of the listed values; applyConditions).
const LA = { "asr.decoder": ["local_agreement"] };
const SEG = { "asr.decoder": ["segmenter"] };

const HINTS = {
  "app.log_level": { label: "Log level", widget: "select", options: ["DEBUG", "INFO", "WARNING", "ERROR"] },
  "audio.device": { label: "Input device", widget: "device",
    help: "\"default\", a microphone (name or part of it), or ndi:<source> to take audio from an NDI source." },
  "audio.sample_rate": { label: "Sample rate (Hz)" },
  "audio.frame_ms": { label: "Frame (ms)" },
  "audio.channels": { label: "Input channels", widget: "intlist",
    help: "1-based channels averaged into mono: 1 = first channel, 1, 2 = stereo downmix." },
  "asr.preset": { label: "Performance preset", widget: "preset" },
  "asr.model": { label: "Whisper model", widget: "datalist", list: "models",
    help: "Pick from the list or type a model name / path to a ggml file." },
  "asr.language_whitelist": { label: "Languages", widget: "languages",
    help: "Empty = detect any language per utterance. One = always that language. Several = the best match per utterance." },
  "asr.unit": { label: "Processing unit", widget: "unit",
    help: "Where the model runs. auto = the recommended unit when the worker starts (CUDA > discrete GPU > CPU > integrated GPU). Picking a GPU also records its name and PCI slot, so a shifted index is reported instead of silently using another card. Never changed by the app itself. Changing it restarts the ASR worker." },
  "asr.unit_identity": { widget: "hidden" },
  "asr.vad_threshold": { label: "VAD speech threshold (0-1)", step: 0.05 },
  "asr.decoder": { label: "Streaming decoder",
    help: "local_agreement: re-decode a growing window and show what consecutive decodes agree on (words trickle out). segmenter: decode each stretch of speech once when it ends (obs-localvocal style); turn partial captions on for a live preview while it is open. Changing it restarts the ASR worker." },
  "asr.step_ms": { label: "Decode step (ms)", when: LA },
  "asr.agreement_passes": { label: "Agreement passes", when: LA },
  "asr.end_silence_ms": { label: "End silence (ms)" },
  "asr.max_utterance_ms": { label: "Max utterance (ms)", when: LA },
  "asr.max_context_seconds": { label: "Max decode window (s)", step: 0.5, when: LA },
  "asr.prune_history_after_silence_seconds": { label: "Prune off-screen history after silence (s)",
    help: "Never touches the visible lines." },
  "asr.wipe_screen_after_silence_seconds": { label: "Wipe screen after silence (s, 0 = off)" },
  "asr.show_partial_captions": { label: "Show partial captions",
    help: "Unconfirmed text shown after the captions. With the segmenter it costs extra decodes of the open segment (GPU time); off = none at all." },
  "asr.vad_preroll_ms": { label: "VAD pre-roll (ms)", when: LA },
  "asr.prompt_carryover_chars": { label: "Prompt carry-over (chars, 0 = off)", when: LA },
  "asr.segmentation_mode": { label: "Segmentation", when: SEG,
    help: "vad: a segment per stretch of speech. hybrid: fixed-length segments, previews only while speech is heard. fixed: fixed-length segments, no VAD (the silence timers never fire)." },
  "asr.partial_latency_ms": { label: "Partial preview every (ms)", when: { ...SEG, "asr.show_partial_captions": [true] },
    help: "Re-decode the open segment for display after this much new audio; grows with the segment's length and gives way when the GPU is busy." },
  "asr.segment_duration_ms": { label: "Segment length (ms)", when: { ...SEG, "asr.segmentation_mode": ["hybrid", "fixed"] } },
  "asr.max_segment_ms": { label: "Longest segment (ms)", when: { ...SEG, "asr.segmentation_mode": ["vad"] },
    help: "Continuous speech is cut here (at a short pause once 60% full) and continues in the next segment; nothing is lost." },
  "asr.context_sentences": { label: "Earlier captions as context (0 = none)", when: SEG,
    help: "The last N captions are given to whisper as the prompt for the next segment; reset by a screen wipe and on a language change." },
  "asr.min_mean_token_p": { label: "Min. token confidence (0-1, 0 = off)", step: 0.05 },
  "asr.max_timestamp_ratio": { label: "Hallucination timestamp ratio (0 = off)", step: 0.25 },
  "asr.strip_bracketed_annotations": { label: "Drop [bracketed] sound annotations" },
  "asr.text_filters": { label: "Text filters (regex find → replace)", widget: "rows",
    help: "Case-insensitive regular expressions applied before anything is captioned; an empty replacement deletes the match." },
  "asr.vocabulary": { label: "Vocabulary (one per line)", widget: "lines",
    help: "Names, places and jargon put at the front of whisper's prompt." },
  "asr.vocabulary_list": { label: "Saved vocabulary list", widget: "vocablist",
    help: "Merged with the vocabulary above. Lists live in <config dir>/vocabulary/." },
  "asr.vocabulary_snap": { label: "Snap near-miss spellings to the vocabulary" },
  "asr.flash_attn": { label: "Flash attention (may crash some GPU drivers)" },
  "asr.dtw_word_timestamps": { label: "Exact word times for transcripts (DTW, slower decodes)",
    help: "Aligns transcript/recording cue times with whisper's DTW; about 20% more decode time. Ignored while flash attention is on." },
  "canvas.width_px": { label: "Width (px)" },
  "canvas.height_px": { label: "Height (px)" },
  "canvas.background_color": { label: "Stream background (RGBA)" },
  "layout.max_lines": { label: "Max lines" },
  "layout.margin_top_pct": { label: "Margin top (%)", step: 0.5 },
  "layout.margin_bottom_pct": { label: "Margin bottom (%)", step: 0.5 },
  "layout.margin_left_pct": { label: "Margin left (%)", step: 0.5 },
  "layout.margin_right_pct": { label: "Margin right (%)", step: 0.5 },
  "layout.interline_pct": { label: "Interline (% of canvas height)", step: 0.1 },
  "layout.vertical_align_within_box": { label: "Vertical align" },
  "layout.pacing": { label: "Typewriter pacing",
    help: "Reveal captions per character, word or sentence (faster when text backs up; a caption arriving alone appears at once). off = show each caption as it arrives." },
  "style.font_family": { label: "Font file", widget: "datalist", list: "fonts" },
  "style.font_color": { label: "Font color (RGBA)" },
  "ndi.source_name": { label: "NDI source name" },
  "ndi.fps": { label: "Frame rate (fps)" },
  "transcript.enabled": { label: "Save a transcript (one file per session)" },
  "transcript.directory": { label: "Folder", widget: "transcriptdir" },
  "captions.blocklist": { label: "Blocklist (one word per line)", widget: "lines",
    help: "Whole words, case- and diacritics-blind; a trailing * matches every word starting with the stem (kurw*)." },
  "captions.blocklist_builtin": { label: "Also use the bundled Polish/English starter list" },
  "captions.blocklist_action": { label: "Blocked words are" },
  "recording.record_on_start": { label: "Start recording when the app starts",
    help: "Recording captures people speaking: get their consent first." },
  "recording.directory": { label: "Folder", help: "Empty = var/recordings in the narrator folder." },
  "recording.retention_days": { label: "Delete recordings older than (days, 0 = never)" },
  "recording.min_free_disk_mb": { label: "Minimum free disk space (MB)",
    help: "Recording will not start, and stops, below this." },
  "recording.on_air_dot": { label: "Red recording dot in the NDI output" },
  "recording.draft_enabled": { label: "Generate a draft transcript after each recording" },
  "recording.draft_model": { label: "Draft model", widget: "datalist", list: "models",
    help: "Offline best-effort pass (large-v3 recommended); it must already be downloaded." },
  "recording.draft_device": { label: "Draft device" },
  "recording.draft_max_asr_busy": { label: "Run drafts next to live ASR up to this load (0-1)", step: 0.05,
    help: "0 = only while live ASR is stopped. Generate draft now ignores it." },
  "web.enabled": { label: "Web UI enabled", help: "Turning this off stops this page; re-enable in config.yaml." },
  "web.bind": { label: "Bind address", help: "Keep 127.0.0.1. Reach it remotely through an SSH tunnel or Tailscale." },
  "web.port": { label: "Port" },
  "web.allowed_hosts": { label: "Extra allowed host names (one per line)", widget: "lines",
    help: "Host headers accepted besides localhost / 127.0.0.1 (DNS-rebinding guard)." },
  "web.preview_fps": { label: "Preview frame rate cap (fps)" },
  "web.shortcuts.clear": { label: "Clear captions", widget: "shortcut" },
  "web.shortcuts.mode_toggle": { label: "Auto / manual switch", widget: "shortcut" },
  "web.shortcuts.focus_entry": { label: "Focus manual entry", widget: "shortcut" },
  "web.shortcuts.submit": { label: "Send manual caption", widget: "shortcut",
    help: "Without Ctrl/Alt/Meta it fires only while the manual entry has focus." },
  "web.shortcuts.pause_toggle": { label: "Pause / resume captions", widget: "shortcut" },
  "web.shortcuts.recording_toggle": { label: "Recording on / off", widget: "shortcut" },
  "web.shortcuts.help": { label: "Show this cheat sheet", widget: "shortcut" },
  "web.api_tokens": { label: "API tokens (Companion, scripts)", widget: "rows", generate: "token",
    help: "Each integration gets its own token for the control API (state, clear, mode, manual text, pause, recording). Delete a row and Save to revoke." },
  "web.osc.enabled": { label: "OSC input enabled", help: "UDP, no authentication: keep it on 127.0.0.1." },
  "web.osc.bind": { label: "OSC bind address" },
  "web.osc.port": { label: "OSC UDP port" },
};

const ACTION_LABELS = {
  clear: "Clear captions", mode_toggle: "Auto / manual switch", focus_entry: "Focus manual entry",
  submit: "Send manual caption", pause_toggle: "Pause / resume captions",
  recording_toggle: "Recording on / off", help: "Show this cheat sheet",
};

function resolve(schema) {
  let s = schema;
  for (let i = 0; s && i < 10; i++) {
    if (s.$ref) s = { ...S.schema.$defs[s.$ref.split("/").pop()], ...withoutRef(s) };
    else if (s.allOf && s.allOf.length === 1) s = { ...s.allOf[0], ...withoutKey(s, "allOf") };
    else break;
  }
  return s || {};
}
function withoutRef(s) { const { $ref, ...rest } = s; return rest; }
function withoutKey(s, key) { const copy = { ...s }; delete copy[key]; return copy; }

function humanize(key) {
  const units = [["_pct", " (%)"], ["_px", " (px)"], ["_ms", " (ms)"], ["_seconds", " (s)"]];
  let suffix = "";
  for (const [end, unit] of units) {
    if (key.endsWith(end)) { key = key.slice(0, -end.length); suffix = unit; break; }
  }
  const words = key.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1) + suffix;
}

function pathId(path) { return path.join("."); }

function onValue(path, value, { rebuild = false } = {}) {
  setPath(S.config, path, value);
  presetToCustomIfOverridden(path, value);
  if (pathId(path) === "asr.model") refreshUnits(true); // the model decides which units fit
  updateDirty();
  applyConditions();
  if (DISPLAY_SECTIONS.includes(path[0])) scheduleDraft();
  if (rebuild) renderTabs();
}

// Editing a field the selected preset covers to a different value means the
// operator is no longer following the preset: show (and save) "custom".
const PRESET_FIELDS = ["model", "step_ms", "agreement_passes"];
function presetToCustomIfOverridden(path, value) {
  if (path.length !== 2 || path[0] !== "asr" || !PRESET_FIELDS.includes(path[1])) return;
  const name = S.config.asr && S.config.asr.preset;
  const p = (S.choices.presets || []).find((x) => x.name === name);
  if (!p || p[path[1]] === value) return;
  S.config.asr.preset = "custom";
  const select = document.getElementById("f-asr.preset");
  if (select) select.value = "custom";
}

function parseNumber(raw, integer) {
  const text = String(raw).trim();
  if (text === "") return raw; // the schema reports it
  const n = Number(text);
  if (!Number.isFinite(n)) return raw;
  if (integer && !Number.isInteger(n)) return raw;
  return n;
}

function renderObject(schema, path, container) {
  const s = resolve(schema);
  for (const [key, propSchema] of Object.entries(s.properties || {})) {
    const childPath = [...path, key];
    const child = resolve(propSchema);
    if (child.properties) {
      const fs = el("fieldset", { "data-path": pathId(childPath) },
        el("legend", { text: SECTION_TITLES[key] || humanize(key) }));
      fs.append(el("div", { class: "err", id: "e-" + pathId(childPath) }));
      renderObject(child, childPath, fs);
      container.append(fs);
    } else {
      container.append(renderField(childPath, child));
    }
  }
}

function widgetFor(id, s, hint) {
  if (hint.widget) return hint.widget;
  if (s.enum) return "enum";
  if (s.anyOf) return "intorauto";
  if (s.type === "boolean") return "checkbox";
  if (s.type === "integer" || s.type === "number") return "number";
  if (s.type === "array") {
    const items = resolve(s.items || {});
    if (items.properties) return "rows";
    if (items.type === "integer") return "intlist";
    return "lines";
  }
  if (id.endsWith("color")) return "rgba";
  return "text";
}

function renderField(path, s) {
  const id = pathId(path);
  const hint = HINTS[id] || {};
  const widget = widgetFor(id, s, hint);
  const value = getPath(S.config, path);
  const inputId = "f-" + id;
  const control = el("div", { class: "control" });
  const wide = ["rows", "lines", "languages"].includes(widget);
  const row = el("div", { class: "field" + (wide ? " wide" : ""), "data-path": id });
  const label = el("label", { for: inputId, text: hint.label || humanize(path[path.length - 1]) });
  const builder = WIDGETS[widget] || WIDGETS.text;
  builder({ path, id, inputId, s, hint, value, control });
  row.append(label, control);
  if (widget === "hidden") row.hidden = true; // set by another widget (asr.unit_identity by the unit picker)
  if (hint.help) row.append(el("div", { class: "help", text: hint.help }));
  row.append(el("div", { class: "err", id: "e-" + id }));
  if (hint.when) row.dataset.when = JSON.stringify(hint.when);
  return row;
}

// Show/hide the rows whose HINTS `when` condition depends on the current
// (unsaved) form values. A row with a validation error always stays visible.
function applyConditions() {
  for (const row of document.querySelectorAll("[data-when]")) {
    const when = JSON.parse(row.dataset.when);
    const ok = Object.entries(when).every(([p, values]) => values.includes(getPath(S.config, p.split("."))));
    const err = row.querySelector(".err");
    row.hidden = !ok && !(err && err.textContent);
  }
}

const WIDGETS = {
  text({ path, inputId, value, control }) {
    control.append(el("input", { type: "text", id: inputId, value: value ?? "",
      oninput: (e) => onValue(path, e.target.value) }));
  },
  number({ path, inputId, value, control, s, hint }) {
    const integer = s.type === "integer";
    control.append(el("input", {
      type: "number", id: inputId, value: value ?? "", step: hint.step ?? (integer ? 1 : "any"),
      min: s.minimum ?? s.exclusiveMinimum, max: s.maximum ?? s.exclusiveMaximum,
      oninput: (e) => onValue(path, parseNumber(e.target.value, integer)),
    }));
  },
  checkbox({ path, inputId, value, control }) {
    control.append(el("input", { type: "checkbox", id: inputId, checked: !!value,
      onchange: (e) => onValue(path, e.target.checked) }));
  },
  enum({ path, inputId, value, control, s }) {
    control.append(selectOf(inputId, s.enum.map((v) => [v, v]), value, (v) => onValue(path, v)));
  },
  select({ path, inputId, value, control, hint }) {
    const options = hint.options.includes(value) ? hint.options : [value, ...hint.options];
    control.append(selectOf(inputId, options.map((v) => [v, v]), value, (v) => onValue(path, v)));
  },
  intorauto({ path, inputId, value, control }) {
    control.append(el("input", { type: "text", id: inputId, value: value ?? "",
      oninput: (e) => {
        const t = e.target.value.trim();
        onValue(path, /^-?\d+$/.test(t) ? Number(t) : t);
      } }));
  },
  datalist({ path, inputId, value, control, hint }) {
    const listId = "dl-" + inputId;
    let options = S.choices[hint.list] || [];
    // The live model list offers only models that run on the chosen unit's kind.
    if (hint.list === "models" && pathId(path) === "asr.model") options = options.filter((m) => modelFitsUnit(m));
    control.append(
      el("input", { type: "text", id: inputId, value: value ?? "", list: listId,
        oninput: (e) => onValue(path, e.target.value) }),
      el("datalist", { id: listId }, options.map((v) => el("option", { value: v }))),
    );
  },
  hidden() {},
  unit({ path, inputId, control }) {
    const select = el("select", { id: inputId, onchange: (e) => {
      const unit = (S.state.units || []).find((u) => u.id === e.target.value);
      // The identity (name + PCI slot) is recorded with the pick; "" for auto and cpu.
      setPath(S.config, ["asr", "unit_identity"], unit && unit.kind !== "cpu" ? unit.identity : "");
      onValue(path, e.target.value);
      refreshUnits(true);
    } });
    const note = el("div", { class: "help", id: "unit-note" });
    const table = el("table", { class: "rec-table units", id: "unit-table" });
    control.append(select, note, table);
    unitsKey = null;
    refreshUnits(true);
  },
  device({ path, inputId, value, control }) {
    const listId = "dl-" + inputId;
    const list = el("datalist", { id: listId }, S.devices.map((v) => el("option", { value: v })));
    control.append(
      el("input", { type: "text", id: inputId, value: value ?? "", list: listId,
        oninput: (e) => onValue(path, e.target.value) }),
      list,
      el("button", { type: "button", text: "Rescan", title: "Re-list microphones and NDI sources",
        onclick: async (e) => {
          e.target.disabled = true;
          await loadDevices(true);
          list.replaceChildren(...S.devices.map((v) => el("option", { value: v })));
          e.target.disabled = false;
          toast(`${S.devices.length} input(s) found`);
        } }),
    );
  },
  intlist({ path, inputId, value, control }) {
    control.append(el("input", { type: "text", id: inputId, value: (value || []).join(", "),
      oninput: (e) => {
        const parts = e.target.value.split(/[,\s]+/).filter(Boolean);
        onValue(path, parts.map((p) => (/^-?\d+$/.test(p) ? Number(p) : p)));
      } }));
  },
  lines({ path, inputId, value, control }) {
    control.append(el("textarea", { id: inputId, rows: 4, value: (value || []).join("\n"),
      oninput: (e) => onValue(path, e.target.value.split("\n").map((l) => l.trim()).filter(Boolean)) }));
  },
  rows({ path, inputId, value, control, s, hint }) {
    const itemSchema = resolve(s.items || {});
    const keys = Object.keys(itemSchema.properties || {});
    const table = el("table", { class: "rows" });
    const draw = () => {
      const rows = getPath(S.config, path) || [];
      table.replaceChildren(
        el("tr", {}, keys.map((k) => el("th", { class: "small muted", text: humanize(k) })), el("th")),
        ...rows.map((row, i) => el("tr", {},
          keys.map((k) => el("td", {}, el("input", { type: "text", value: row[k] ?? "",
            "aria-label": `${humanize(k)} ${i + 1}`,
            oninput: (e) => { row[k] = e.target.value; onValue(path, rows); } }))),
          el("td", {}, el("button", { type: "button", text: "Remove",
            onclick: () => { rows.splice(i, 1); onValue(path, rows); draw(); } })),
        )),
      );
    };
    const add = el("button", { type: "button", id: inputId, text: hint.generate ? "Add token" : "Add rule",
      onclick: () => {
        const rows = getPath(S.config, path) || [];
        const fresh = Object.fromEntries(keys.map((k) => [k, ""]));
        if (hint.generate === "token") {
          fresh.name = `integration-${rows.length + 1}`;
          fresh.token = randomToken();
        }
        rows.push(fresh);
        onValue(path, rows);
        draw();
      } });
    draw();
    control.append(table, add);
  },
  languages({ path, inputId, value, control }) {
    const names = Object.fromEntries(S.choices.languages);
    const chips = el("div", { class: "chips" });
    const draw = () => {
      const codes = getPath(S.config, path) || [];
      chips.replaceChildren(
        ...(codes.length ? codes.map((code, i) => el("span", { class: "chip" }, names[code] || code,
          el("button", { type: "button", text: "×", "aria-label": `Remove ${names[code] || code}`,
            onclick: () => { codes.splice(i, 1); onValue(path, codes); draw(); } })))
          : [el("span", { class: "muted small", text: "Auto-detect (no restriction)" })]),
      );
    };
    const picker = selectOf(inputId, [["", "Add a language…"], ...S.choices.languages], "", () => {});
    const add = el("button", { type: "button", text: "Add", onclick: () => {
      const code = picker.value;
      const codes = getPath(S.config, path) || [];
      if (code && !codes.includes(code)) { codes.push(code); onValue(path, codes); draw(); }
      picker.value = "";
    } });
    draw();
    control.append(chips, picker, add);
  },
  rgba({ path, inputId, value, control }) {
    const text = el("input", { type: "text", id: inputId, value: value ?? "", class: "rgba-text",
      oninput: (e) => { onValue(path, e.target.value); syncPickers(); } });
    const color = el("input", { type: "color", class: "swatch", "aria-label": "Color" });
    const alpha = el("input", { type: "range", min: 0, max: 255, class: "alpha", "aria-label": "Opacity" });
    const syncPickers = () => {
      const m = /^#([0-9a-f]{6})([0-9a-f]{2})$/i.exec(text.value);
      if (m) { color.value = "#" + m[1].toLowerCase(); alpha.value = parseInt(m[2], 16); }
    };
    const fromPickers = () => {
      const a = Number(alpha.value).toString(16).padStart(2, "0").toUpperCase();
      text.value = color.value.toUpperCase() + a;
      onValue(path, text.value);
    };
    color.addEventListener("input", fromPickers);
    alpha.addEventListener("input", fromPickers);
    syncPickers();
    control.append(color, alpha, text);
  },
  shortcut({ path, inputId, value, control }) {
    const input = el("input", { type: "text", id: inputId, value: value ?? "", class: "shortcut",
      readOnly: true, "data-capture": "1", title: "Click, then press the key combination (Backspace = unbind)" });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Tab") return;
      e.preventDefault();
      e.stopPropagation();
      if ((e.key === "Backspace" || e.key === "Delete") && !e.ctrlKey && !e.altKey && !e.metaKey) {
        input.value = "";
      } else {
        const combo = comboFromEvent(e);
        if (!combo) return;
        input.value = combo;
      }
      onValue(path, input.value);
    });
    control.append(input, el("button", { type: "button", text: "Unbind",
      onclick: () => { input.value = ""; onValue(path, ""); } }));
  },
  preset({ path, inputId, value, control }) {
    const presets = S.choices.presets || [];
    const options = [["custom", "Custom (no preset)"],
      ...presets.map((p) => [p.name, `${p.name} — ${p.model}, step ${p.step_ms} ms${p.measured ? "" : " (provisional)"}`])];
    const note = el("div", { class: "help" });
    const showNote = (name) => {
      const p = presets.find((x) => x.name === name);
      note.textContent = p ? p.note : "Model, decode step and agreement passes are set individually below.";
    };
    const select = selectOf(inputId, options, value ?? "custom", (name) => {
      const p = presets.find((x) => x.name === name);
      setPath(S.config, path, name);
      if (p) {
        // The preset supplies these; Save then leaves them out of config.yaml
        // so the file keeps following the preset (config.loader.save_config).
        for (const k of ["model", "step_ms", "agreement_passes"]) setPath(S.config, [...path.slice(0, -1), k], p[k]);
      }
      onValue(path, name, { rebuild: true });
    });
    const bench = el("button", { type: "button", id: "bench-btn", text: "Re-run benchmark",
      title: "narrator --pick-preset: measures this machine and writes the best preset to config.yaml",
      onclick: runBenchmark });
    const benchStatus = el("span", { class: "small muted", id: "bench-status" });
    showNote(value);
    control.append(select, bench, benchStatus, note);
    updateBenchmark();
  },
  vocablist({ path, inputId, value, control }) {
    const listId = "dl-" + inputId;
    const editor = el("div", { class: "vocab-editor", hidden: true });
    const input = el("input", { type: "text", id: inputId, value: value ?? "", list: listId,
      oninput: (e) => onValue(path, e.target.value) });
    control.append(
      input,
      el("datalist", { id: listId }, S.choices.vocabulary_lists.map((v) => el("option", { value: v }))),
      el("button", { type: "button", text: "Edit list…", onclick: () => openVocabEditor(input.value.trim(), editor) }),
      editor,
    );
  },
  transcriptdir({ path, inputId, value, control }) {
    control.append(el("input", { type: "text", id: inputId, value: value ?? "",
      placeholder: S.choices.transcript_default_dir || "", oninput: (e) => onValue(path, e.target.value) }));
    if (S.choices.transcript_default_dir) {
      control.append(el("span", { class: "small muted", text: `Empty = ${S.choices.transcript_default_dir}` }));
    }
  },
};

// ------------------------------------------------------------------ processing units

// Model catalog (narrator.asr.catalog): unit kinds a model runs on. A model
// the catalog does not list is a whisper.cpp ggml model (cpu, cuda, vulkan).
function catalogUnits(model) {
  const entry = (S.choices.catalog || []).find((c) => c.model === model);
  return entry ? entry.units : ["cpu", "cuda", "vulkan"];
}

function unitKind(unit) { return String(unit || "auto").split(":")[0]; }

function modelFitsUnit(model) {
  const kind = unitKind(getPath(S.config, ["asr", "unit"]));
  return kind === "auto" || catalogUnits(model).includes(kind);
}

const fmt = (v, unit, digits = 0) => (v === null || v === undefined ? "n/a" : `${Number(v).toFixed(digits)}${unit}`);

function memText(u) {
  const m = u.usage || {};
  if (m.mem_used_mb === null || m.mem_used_mb === undefined) return m.mem_total_mb ? `n/a / ${fmt(m.mem_total_mb / 1024, " GB", 1)}` : "n/a";
  return `${fmt(m.mem_used_mb / 1024, "", 1)} / ${fmt(m.mem_total_mb / 1024, " GB", 1)}`;
}

function unitOptionLabel(u) {
  const vram = u.kind !== "cpu" && u.type === "discrete" && u.vram_mb ? `, ${(u.vram_mb / 1024).toFixed(0)} GB` : "";
  const tags = [u.recommended ? "recommended" : "", u.type === "integrated" ? "integrated" : ""].filter(Boolean);
  return `${u.id} — ${u.name} (${u.backend}${vram})${tags.length ? " · " + tags.join(", ") : ""}${u.usable ? "" : " · unavailable"}`;
}

// Re-render the unit picker's options (only when the unit list, the model
// or the selection changed, so an open dropdown is not replaced) and the
// live usage table (every state message).
let unitsKey = null;
function refreshUnits(force = false) {
  const select = document.getElementById("f-asr.unit");
  if (!select || !S.config) return;
  const units = S.state.units || [];
  const value = getPath(S.config, ["asr", "unit"]) || "auto";
  const kinds = catalogUnits(getPath(S.config, ["asr", "model"]));
  const key = JSON.stringify([units.map((u) => [u.id, u.usable, u.recommended]), value, kinds]);
  if (force || key !== unitsKey) {
    unitsKey = key;
    const rec = units.find((u) => u.recommended);
    const options = [el("option", { value: "auto", text: `auto — the recommended unit${rec ? ` (now ${rec.id}, ${rec.name})` : ""}` })];
    for (const u of units) {
      const fits = kinds.includes(u.kind);
      options.push(el("option", { value: u.id, text: unitOptionLabel(u) + (fits ? "" : " · not for this model"),
        disabled: (!u.usable || !fits) && u.id !== value, title: u.reason || (fits ? "" : `the model runs on ${kinds.join(", ")}`) }));
    }
    if (value !== "auto" && !units.some((u) => u.id === value)) {
      options.push(el("option", { value, text: units.length ? `${value} — not on this machine` : value }));
    }
    select.replaceChildren(...options);
    select.value = value;
  }
  const note = document.getElementById("unit-note");
  if (note) {
    const chosen = units.find((u) => u.id === value);
    const identity = getPath(S.config, ["asr", "unit_identity"]) || "";
    const parts = [];
    if (value === "auto") parts.push(S.state.unit_resolved ? `auto runs on ${S.state.unit_resolved}.` : "auto: no usable unit found.");
    if (chosen && !chosen.usable) parts.push(`Not usable: ${chosen.reason}.`);
    if (chosen && chosen.kind !== "cpu") parts.push(identity ? `Recorded card: ${identity}.` : "No card recorded yet (pick the unit again to record it).");
    if (!units.length) parts.push("Listing units…");
    note.textContent = parts.join(" ");
  }
  const table = document.getElementById("unit-table");
  if (table) {
    const head = el("tr", {}, ["Unit", "Device", "Load", "Memory", "Temp", "Power", ""].map((h) => el("th", { text: h })));
    table.replaceChildren(el("thead", {}, head), el("tbody", {}, units.map((u) => {
      const m = u.usage || {};
      const cores = Array.isArray(m.per_core_pct) && m.per_core_pct.length
        ? ` (${m.per_core_pct.map((c) => fmt(c, "%")).join(" ")})` : "";
      const status = [u.selected ? "in use" : "", u.recommended ? "recommended" : "", u.usable ? "" : (u.reason || "unavailable")]
        .filter(Boolean).join(" · ");
      return el("tr", { class: u.selected ? "selected" : "" },
        el("td", { text: u.id }),
        el("td", { text: `${u.name} (${u.backend}${u.pci_bus_id ? ", " + u.pci_bus_id : ""})` }),
        el("td", { text: fmt(m.util_pct, "%") + cores }),
        el("td", { text: memText(u) }),
        el("td", { text: fmt(m.temp_c, " °C") }),
        el("td", { text: fmt(m.power_w, " W", 1) }),
        el("td", { class: "small muted", text: status }));
    })));
  }
}

function selectOf(id, options, value, onChange) {
  const select = el("select", { id, onchange: (e) => onChange(e.target.value) },
    options.map(([v, label]) => el("option", { value: v, text: label })));
  select.value = value;
  return select;
}

function randomToken() {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function openVocabEditor(name, editor) {
  if (!name) { toast("Type a list name first", true); return; }
  const res = await api("GET", `/api/vocabulary/${encodeURIComponent(name)}`);
  if (!res.ok) { toast(res.data?.error || "Cannot load the list", true); return; }
  const area = el("textarea", { rows: 6, value: res.data.entries.join("\n"), "aria-label": `Vocabulary list ${name}` });
  editor.replaceChildren(
    el("div", { class: "small muted", text: `List "${name}" (one entry per line; saved as its own file, applied live):` }),
    area,
    el("button", { type: "button", text: "Save list", onclick: async () => {
      const entries = area.value.split("\n").map((l) => l.trim()).filter(Boolean);
      const put = await api("PUT", `/api/vocabulary/${encodeURIComponent(name)}`, { entries });
      if (put.ok) {
        toast(`Saved list "${name}"`);
        if (!S.choices.vocabulary_lists.includes(name)) S.choices.vocabulary_lists.push(name);
      } else toast(put.data?.error || "Saving the list failed", true);
    } }),
    el("button", { type: "button", class: "ghost", text: "Close", onclick: () => { editor.hidden = true; } }),
  );
  editor.hidden = false;
}

// ------------------------------------------------------------------ tabs

function renderTabs() {
  const nav = $("#tabs");
  const panels = $("#tab-panels");
  const scroll = panels.scrollTop;
  const known = new Set(TABS.flatMap((t) => t.sections));
  const extra = Object.keys(S.schema.properties).filter((k) => !known.has(k));
  const tabs = extra.length ? [...TABS.slice(0, -1), { id: "other", title: "Other", sections: extra }, TABS[TABS.length - 1]] : TABS;
  if (!S.activeTab) S.activeTab = tabs[0].id;
  nav.replaceChildren(...tabs.map((t) => el("button", {
    type: "button", role: "tab", id: "tab-" + t.id, text: t.title, "aria-selected": String(t.id === S.activeTab),
    onclick: () => { S.activeTab = t.id; renderTabs(); },
  })));
  const tab = tabs.find((t) => t.id === S.activeTab) || tabs[0];
  const panel = el("div", { role: "tabpanel", "aria-labelledby": "tab-" + tab.id });
  if (tab.id === "status") panel.append(buildStatus());
  if (tab.id === "recordings") panel.append(buildRecordings());
  for (const section of tab.sections) {
    const schema = S.schema.properties[section];
    if (!schema) continue;
    const fs = el("fieldset", { "data-path": section }, el("legend", { text: SECTION_TITLES[section] || humanize(section) }));
    fs.append(el("div", { class: "err", id: "e-" + section }));
    renderObject(schema, [section], fs);
    panel.append(fs);
  }
  if (tab.id === "remote") panel.append(buildRemoteHelp());
  panels.replaceChildren(panel);
  panels.scrollTop = scroll;
  showErrors();
  applyConditions();
  if (tab.id === "status") renderLog();
  if (tab.id === "recordings") refreshRecordings();
}

function buildRemoteHelp() {
  return el("fieldset", {}, el("legend", { text: "Control API" }),
    el("p", { class: "small", text: "Endpoints (POST, JSON body or query parameters; Authorization: Bearer <API token>):" }),
    el("pre", { class: "log", text: [
      "GET  /api/state",
      "POST /api/clear",
      "POST /api/mode          {\"mode\": \"auto\" | \"manual\" | \"toggle\"}",
      "POST /api/manual-text   {\"text\": \"...\"}",
      "POST /api/pause         {\"paused\": true | false | \"toggle\"}",
      "POST /api/recording     {\"recording\": true | false | \"toggle\"}",
      "",
      "OSC: /narrator/clear, /narrator/mode [s], /narrator/manual-text s,",
      "     /narrator/pause [i|T|F], /narrator/recording [i|T|F]",
    ].join("\n") }),
    el("p", { class: "small muted", text: "Bitfocus Companion: see the README section \"Web UI and remote control\" for a ready button set." }));
}

// ------------------------------------------------------------------ recordings tab

function formatBytes(n) {
  if (!Number.isFinite(n)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function buildRecordings() {
  return el("fieldset", { id: "recordings-box" }, el("legend", { text: "Recordings" }),
    el("p", { class: "small muted", text: "Each session keeps the audio the recognizer heard and the captions that went on air; " +
      "a draft transcript is made afterwards. Download, correct and upload the SRT back; a corrected recording can join " +
      "the reference library that tuning learns from. Recordings never leave this machine unless downloaded." }),
    el("div", { class: "rec-tools" },
      el("span", { id: "rec-disk", class: "small muted" }),
      el("span", { class: "spacer" }),
      el("button", { type: "button", class: "danger", id: "rec-delete-all", text: "Delete all recordings…", onclick: deleteAllRecordings })),
    el("div", { id: "rec-list" }, el("p", { class: "small muted", text: "Loading…" })));
}

let recTimer = null;
async function refreshRecordings() {
  clearTimeout(recTimer);
  if (S.activeTab !== "recordings" || !$("#rec-list")) return;
  const res = await api("GET", "/api/recordings");
  if (S.activeTab !== "recordings" || !$("#rec-list")) return;
  if (!res.ok) {
    $("#rec-list").replaceChildren(el("p", { class: "err", text: res.data?.error || `Cannot list recordings (${res.status})` }));
  } else renderRecordings(res.data);
  recTimer = setTimeout(refreshRecordings, 3000);
}

function renderRecordings(data) {
  const d = data.disk || {};
  $("#rec-disk").textContent = `Recordings use ${formatBytes(d.recordings_bytes)}; ${formatBytes(d.free_bytes)} free ` +
    `(recording stops below ${d.min_free_disk_mb} MB) — ${d.directory || ""}`;
  const list = data.recordings || [];
  $("#rec-delete-all").disabled = !list.some((r) => r.status !== "recording");
  if (!list.length) {
    $("#rec-list").replaceChildren(el("p", { class: "small muted", text: "No recordings yet. Use Record (or its shortcut) to start one." }));
    return;
  }
  const head = el("thead", {}, el("tr", {}, ["Started", "Duration", "Size", "Status", ""].map((h) => el("th", { text: h }))));
  const body = el("tbody", {}, list.map(recordingRow));
  $("#rec-list").replaceChildren(el("table", { class: "rec-table" }, head, body));
}

function recordingRow(r) {
  const url = (name) => `/api/recordings/${encodeURIComponent(r.id)}/files/${name}`;
  const live = r.status === "recording";
  const started = r.started_at ? new Date(r.started_at).toLocaleString() : r.id;
  const duration = live ? formatDuration(S.state.recording_info?.elapsed_s || 0) : formatDuration(r.duration_s || 0);
  let detail = "";
  if (r.status === "draft pending") {
    detail = r.draft.state === "running" ? (r.draft.suspended ? "draft paused while live ASR is busy" : "generating draft…")
      : (r.draft.waiting || "draft queued");
  } else if (r.status === "draft failed") detail = r.draft.error || "";
  if (r.dropped_frames) detail += `${detail ? "; " : ""}${r.dropped_frames} audio frame(s) dropped`;
  if (r.session_status && !["complete", "recording"].includes(r.session_status)) detail += `${detail ? "; " : ""}${r.stop_reason || r.session_status}`;
  if (r.reference) detail += `${detail ? "; " : ""}in the reference library`;
  const actions = el("div", { class: "actions" });
  const dl = (name, label) => r.files[name] && actions.append(el("a", { class: "btn", href: url(name), download: "", text: label }));
  dl("audio.flac", "Audio");
  dl("live.srt", "Live SRT");
  dl("draft.srt", "Draft SRT");
  dl("corrected.srt", "Corrected SRT");
  if (!live) {
    const file = el("input", { type: "file", accept: ".srt,text/plain", hidden: true,
      onchange: (e) => uploadCorrected(r.id, e.target.files[0]) });
    actions.append(file,
      el("button", { type: "button", text: "Upload corrected…", onclick: () => file.click() }),
      el("button", { type: "button", text: r.files["draft.srt"] ? "Redo draft" : "Generate draft now",
        disabled: r.draft.state === "running" && !r.draft.suspended,
        onclick: () => recAction("POST", `/api/recordings/${encodeURIComponent(r.id)}/draft`, "Draft requested") }),
      el("button", { type: "button", text: "Use as reference", disabled: !r.files["corrected.srt"],
        title: r.files["corrected.srt"] ? "Copy into scripts/test_audio for narrator-tune" : "Upload a corrected SRT first",
        onclick: () => recAction("POST", `/api/recordings/${encodeURIComponent(r.id)}/reference`, "Copied into the reference library") }),
      el("button", { type: "button", class: "danger", text: "Delete", onclick: () => {
        if (confirm(`Delete the recording from ${started}? This cannot be undone.`)) {
          recAction("DELETE", `/api/recordings/${encodeURIComponent(r.id)}`, "Recording deleted");
        }
      } }));
  }
  const tag = r.status.replace(/\s+/g, "-");
  return el("tr", {},
    el("td", { text: started }),
    el("td", { text: duration }),
    el("td", { text: formatBytes(r.size_bytes) }),
    el("td", {}, el("span", { class: `status-tag ${tag}`, text: r.status }), detail ? el("div", { class: "small muted", text: detail }) : null),
    el("td", {}, actions));
}

async function recAction(method, path, okMessage, body) {
  const res = await api(method, path, body);
  if (res.ok) toast(okMessage);
  else toast(res.data?.error || `Failed (${res.status})`, true);
  refreshRecordings();
}

async function uploadCorrected(id, file) {
  if (!file) return;
  const res = await fetch(`/api/recordings/${encodeURIComponent(id)}/corrected`, {
    method: "PUT", credentials: "same-origin", headers: { "Content-Type": "application/x-subrip" }, body: file,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (res.ok) toast(`Corrected SRT saved (${data.cues} cues)`);
  else toast(data?.error || `Upload failed (${res.status})`, true);
  refreshRecordings();
}

async function deleteAllRecordings() {
  if (!confirm("Delete ALL recordings (audio and subtitles)? A recording that is still running is kept. This cannot be undone.")) return;
  await recAction("POST", "/api/recordings/delete-all", "All recordings deleted", { confirm: true });
}

// ------------------------------------------------------------------ status tab

function buildStatus() {
  const st = S.state;
  const dl = el("dl", { class: "status-grid", id: "status-grid" });
  const rows = [
    ["ASR worker", st.asr_active ? "Active" : "Inactive"],
    ["ASR backend", st.asr_backend || "unresolved"],
    ["Worker restarts (config)", st.asr_restarts ?? 0],
    ["Worker crashes", `${st.asr_crashes ?? 0}${st.asr_last_exit_code !== null && st.asr_last_exit_code !== undefined ? ` (last exit code ${st.asr_last_exit_code})` : ""}`],
    ["Caption mode", `${st.caption_mode || "auto"}${st.captions_paused ? ", paused" : ""}`],
    ["ASR load", st.asr_busy !== null && st.asr_busy !== undefined ? `${Math.round(st.asr_busy * 100)}% busy decoding` : "-"],
    ["Recording", st.recording ? `on (${st.recording_info?.session_id || ""})` : "off"],
    ["Uptime", st.uptime_s !== null && st.uptime_s !== undefined ? formatDuration(st.uptime_s) : "-"],
    ["Config file", S.configPath || "-"],
  ];
  for (const [k, v] of rows) dl.append(el("dt", { text: k }), el("dd", { text: String(v) }));
  const level = el("select", { id: "log-level", "aria-label": "Minimum log level", onchange: renderLog },
    ["DEBUG", "INFO", "WARNING", "ERROR"].map((l) => el("option", { value: l, text: `${l} and up` })));
  level.value = S.logLevel || "INFO";
  return el("div", {},
    el("fieldset", {}, el("legend", { text: "Status" }), dl,
      el("button", { type: "button", class: "ghost", text: "Log out", onclick: async () => {
        await api("POST", "/api/logout");
        location.href = "/";
      } })),
    el("fieldset", {}, el("legend", { text: "Recent log" }),
      el("div", { class: "log-tools" }, level,
        el("label", { class: "small" }, el("input", { type: "checkbox", id: "log-follow", checked: true }), " Follow")),
      el("div", { class: "log", id: "log", "aria-live": "off" })));
}

function formatDuration(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

const LEVELS = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };

function renderLog() {
  const box = $("#log");
  if (!box) return;
  const select = $("#log-level");
  S.logLevel = select ? select.value : "INFO";
  const min = LEVELS[S.logLevel] || 0;
  const lines = S.logLines.filter((l) => (LEVELS[l.level] || 0) >= min).slice(-600);
  box.replaceChildren(...lines.map(logLine));
  if ($("#log-follow")?.checked) box.scrollTop = box.scrollHeight;
}

function logLine(l) {
  const t = new Date(l.time * 1000).toLocaleTimeString();
  return el("div", { class: l.level, text: `${t} ${l.level.padEnd(7)} ${l.message}` });
}

function appendLog(lines) {
  S.logLines.push(...lines);
  if (S.logLines.length > 1500) S.logLines.splice(0, S.logLines.length - 1500);
  const box = $("#log");
  if (!box) return;
  const min = LEVELS[S.logLevel || "INFO"] || 0;
  for (const l of lines) if ((LEVELS[l.level] || 0) >= min) box.append(logLine(l));
  while (box.childElementCount > 600) box.firstChild.remove();
  if ($("#log-follow")?.checked) box.scrollTop = box.scrollHeight;
}

// ------------------------------------------------------------------ config load/save

async function loadConfig() {
  const res = await api("GET", "/api/config");
  S.saved = res.data.config;
  S.config = clone(S.saved);
  S.configPath = res.data.config_path;
  S.rev = res.data.rev;
  S.errors = {};
  const banner = $("#banner");
  if (res.data.file_error) {
    banner.textContent = `config.yaml is currently invalid and was not applied; showing the live config. Saving replaces the file. (${res.data.file_error})`;
    banner.hidden = false;
  } else banner.hidden = true;
  renderTabs();
  updateDirty();
  scheduleDraft();
}

function isDirty() { return S.config && !same(S.config, S.saved); }

function updateDirty() {
  const dirty = isDirty();
  $("#save-btn").disabled = !dirty;
  $("#revert-btn").disabled = !dirty;
  $("#dirty").textContent = dirty ? "Unsaved changes" : "No unsaved changes";
  $(".savebar").classList.toggle("dirty", dirty);
}

async function saveConfig() {
  const button = $("#save-btn");
  button.disabled = true;
  const res = await api("POST", "/api/config", { config: S.config });
  if (res.status === 422) {
    S.errors = Object.fromEntries(res.data.errors.map((e) => [e.loc, e.msg]));
    showErrors();
    toast(`Not saved: ${res.data.errors.length} problem(s)`, true);
    updateDirty();
    return;
  }
  if (!res.ok) { toast(res.data?.error || `Save failed (${res.status})`, true); updateDirty(); return; }
  S.errors = {};
  S.saved = res.data.config;
  S.config = clone(S.saved);
  renderTabs();
  updateDirty();
  scheduleDraft();
  toast(res.data.changed ? "Saved — the running app applies it now" : "Nothing changed");
  if (res.data.new_url) {
    toast(`Web UI moves to ${res.data.new_url}`);
    setTimeout(() => { location.href = res.data.new_url; }, 1500);
  }
}

function fieldFor(loc) {
  // The longest prefix of the error location that has a field/fieldset.
  const parts = loc.split(".");
  for (let n = parts.length; n > 0; n--) {
    const id = parts.slice(0, n).join(".");
    if (document.getElementById("e-" + id)) return id;
  }
  return null;
}

function tabFor(loc) {
  const section = loc.split(".")[0];
  const tab = TABS.find((t) => t.sections.includes(section));
  return tab ? tab.id : (S.schema.properties[section] ? "other" : null);
}

function showErrors() {
  document.querySelectorAll(".field.invalid").forEach((f) => f.classList.remove("invalid"));
  document.querySelectorAll("[id^='e-']").forEach((e) => { e.textContent = ""; });
  const summary = $("#error-summary");
  const entries = Object.entries(S.errors);
  const tabsWithErrors = new Set(entries.map(([loc]) => tabFor(loc)));
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("has-error", tabsWithErrors.has(b.id.slice(4)));
  });
  for (const [loc, msg] of entries) {
    const id = fieldFor(loc);
    if (!id) continue;
    const box = document.getElementById("e-" + id);
    const rest = loc.slice(id.length + 1);
    box.textContent = (box.textContent ? box.textContent + " • " : "") + (rest ? `${rest}: ${msg}` : msg);
    box.closest(".field")?.classList.add("invalid");
  }
  summary.hidden = entries.length === 0;
  applyConditions();
  summary.replaceChildren(...entries.map(([loc, msg]) => el("li", {},
    el("a", { text: loc || "config", onclick: () => {
      const tab = tabFor(loc);
      if (tab) { S.activeTab = tab; renderTabs(); }
      const id = fieldFor(loc);
      const input = id && document.getElementById("f-" + id);
      (input || document.getElementById("e-" + id))?.scrollIntoView({ block: "center" });
      input?.focus();
    } }), `: ${msg}`)));
}

// ------------------------------------------------------------------ preview drafts

let draftTimer = null;
function scheduleDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(sendDraft, 150);
  updateGuides();
}

function displayOf(cfg) { return Object.fromEntries(DISPLAY_SECTIONS.map((k) => [k, cfg[k]])); }

function sendDraft() {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN || !S.config) return;
  const draft = displayOf(S.config);
  const unsaved = !same(draft, displayOf(S.saved));
  S.ws.send(JSON.stringify({ type: "draft", display: unsaved ? draft : null }));
  $("#preview-note").textContent = unsaved ? "Previewing unsaved display settings (not on air)" : "";
}

function updateGuides() {
  const cfg = S.config;
  if (!cfg) return;
  const box = $("#preview-box");
  const w = Number(cfg.canvas?.width_px), h = Number(cfg.canvas?.height_px);
  if (w > 0 && h > 0) box.style.aspectRatio = `${w} / ${h}`;
  const g = $("#guides");
  const l = cfg.layout || {};
  const nums = [l.margin_top_pct, l.margin_right_pct, l.margin_bottom_pct, l.margin_left_pct].map(Number);
  if (!$("#guides-toggle").checked || nums.some((n) => !Number.isFinite(n))) { g.hidden = true; return; }
  g.hidden = false;
  [g.style.top, g.style.right, g.style.bottom, g.style.left] = nums.map((n) => `${n}%`);
}

// ------------------------------------------------------------------ live state

function updateState(st) {
  S.state = st;
  S.shortcuts = st.shortcuts || S.shortcuts;
  const asr = $("#asr-pill");
  asr.textContent = st.asr_active ? "ASR active" : (st.asr_halted ? "ASR stopped" : "ASR inactive");
  asr.className = "pill " + (st.asr_active ? "ok" : "bad");
  renderProblems(st.problems || []);
  $("#backend").textContent = st.asr_backend && st.asr_backend !== "unresolved" ? st.asr_backend : "";
  const manual = st.caption_mode === "manual";
  const mode = $("#mode-pill");
  mode.textContent = manual ? "Manual" : "Auto";
  mode.className = "pill" + (manual ? " manual" : "");
  document.body.classList.toggle("mode-manual", manual);
  $("#mode-auto").setAttribute("aria-pressed", String(!manual));
  $("#mode-manual").setAttribute("aria-pressed", String(manual));
  $("#paused-pill").hidden = !st.captions_paused;
  const pause = $("#pause-btn");
  pause.textContent = st.captions_paused ? "Resume captions" : "Pause captions";
  pause.classList.toggle("active", !!st.captions_paused);
  const rec = $("#rec-btn");
  rec.disabled = !st.recording_available;
  rec.title = st.recording_available ? "Record the audio and on-air captions of this session" : "Recording is not available (needs the soundfile package)";
  rec.textContent = st.recording ? "Stop recording" : "Record";
  rec.classList.toggle("active", !!st.recording);
  const recPill = $("#rec-pill");
  recPill.hidden = !st.recording;
  const info = st.recording_info || {};
  recPill.textContent = st.recording ? `REC ${formatDuration(info.elapsed_s || 0)}` : "REC";
  if (S.wasRecording && !st.recording && info.last_stop_reason && info.last_stop_reason !== "stopped by the operator") {
    toast(`Recording stopped: ${info.last_stop_reason}`, true);
  }
  S.wasRecording = !!st.recording;
  $("#manual-hint").textContent = manual
    ? "Manual mode: ASR output is not put on air."
    : (st.captions_paused ? "Captions paused: ASR keeps listening, nothing new goes on air." : "Auto mode: lines you send are added after the ASR captions.");
  const level = Math.max(0, Math.min(1, st.mic_level || 0));
  const bar = $("#meter-bar");
  bar.style.width = `${Math.pow(level, 1 / 3) * 100}%`;
  bar.classList.toggle("clip", level >= 0.98);
  const onAir = $("#on-air");
  const lines = st.on_air || [];
  if (onAir.dataset.lines !== JSON.stringify(lines)) {
    onAir.dataset.lines = JSON.stringify(lines);
    onAir.replaceChildren(...lines.map((line) => el("span", { class: "line", text: line })));
  }
  if (S.rev !== null && st.config_rev !== S.rev) {
    S.rev = st.config_rev;
    if (!isDirty()) loadConfig();
    else {
      const banner = $("#banner");
      banner.textContent = "The config was changed elsewhere (file edit or another tab). Revert to load it, or Save to overwrite it.";
      banner.hidden = false;
    }
  }
  updateBenchmark();
  refreshUnits();
  if (S.activeTab === "status") {
    const grid = $("#status-grid");
    if (grid) grid.replaceWith(buildStatus().querySelector("#status-grid"));
  }
}

// ------------------------------------------------------------------ problems panel

// Setup checks (narrator.health): one row per problem, most severe first.
// Fix buttons: "ui:<tab>" opens that settings tab, anything else is POSTed
// to /api/fix (the server restarts only the affected part). Re-rendered
// only when the list changes, so a button is not replaced under the cursor.
const SUBSYSTEM_TITLES = { asr: "ASR", audio: "Audio", ndi: "NDI", web: "Web", recording: "Recording", setup: "Setup" };
let problemsKey = null;

function renderProblems(problems) {
  const key = JSON.stringify(problems);
  if (key === problemsKey) return;
  problemsKey = key;
  const box = $("#problems");
  box.hidden = problems.length === 0;
  box.className = "problems" + (problems.some((p) => p.severity === "error") ? " has-error" : "");
  $("#problem-list").replaceChildren(...problems.map((p) => el("li", { class: "problem " + p.severity },
    el("span", { class: "problem-tag", text: `${SUBSYSTEM_TITLES[p.subsystem] || p.subsystem} ${p.severity}` }),
    el("span", { class: "problem-msg", text: p.message }),
    el("span", { class: "problem-fixes" }, ...(p.fixes || []).map((f) => el("button", {
      type: "button", text: f.label, "data-action": f.action, onclick: (e) => runFix(f, e.currentTarget),
    }))),
  )));
}

async function runFix(fix, button) {
  if (fix.action.startsWith("ui:")) {
    S.activeTab = fix.action.slice(3);
    renderTabs();
    $("#tabs").scrollIntoView({ block: "nearest" });
    return;
  }
  if (button) button.disabled = true;
  const res = await control("/api/fix", { action: fix.action }, `${fix.label}: done`);
  if (!res && button) button.disabled = false;
}

function updateBenchmark() {
  const b = S.state.benchmark || {};
  const btn = $("#bench-btn"), status = $("#bench-status");
  if (!btn) return;
  btn.disabled = !!b.running;
  status.textContent = b.running ? "Benchmark running… (progress in Status & log)"
    : b.exit_code === null || b.exit_code === undefined ? ""
    : b.exit_code === 0 ? "Last benchmark: done (config updated if a preset was chosen)"
    : `Last benchmark: exit code ${b.exit_code} (see Status & log)`;
}

async function runBenchmark() {
  if (!confirm("Benchmark this machine and pick the most accurate preset it can carry?\n\n" +
    "It runs for about a minute and loads models on the GPU next to the live worker, so live captions can lag meanwhile. " +
    "The chosen preset is written to config.yaml (previous file kept as config.yaml.bak).")) return;
  const res = await api("POST", "/api/preset/benchmark");
  if (res.ok) toast("Benchmark started");
  else toast(res.data?.error || "Could not start the benchmark", true);
}

function addFeed(events) {
  const feed = $("#feed");
  for (const ev of events) {
    if (ev.kind === "clear") continue; // off-screen pruning: nothing visible changed
    const last = feed.lastElementChild;
    if (last && last.classList.contains("partial")) last.remove();
    let item;
    if (ev.kind === "final" || ev.kind === "partial") {
      item = el("li", { class: ev.kind + (ev.applied ? "" : " ignored") }, ev.text);
      if (!ev.applied) item.append(el("span", { class: "tag", text: "not on air" }));
    } else if (ev.kind === "wipe") {
      item = el("li", { class: "system", text: "— screen wiped after silence —" });
    } else if (ev.kind === "status") {
      item = el("li", { class: "system", text: `ASR backend: ${ev.text}` });
    } else continue;
    feed.append(item);
  }
  while (feed.childElementCount > 80) feed.firstChild.remove();
  feed.scrollTop = feed.scrollHeight;
}

// ------------------------------------------------------------------ websocket

let previewUrl = null;
let wsRetry = 500;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "blob";
  S.ws = ws;
  ws.onopen = () => {
    wsRetry = 500;
    const conn = $("#conn");
    conn.textContent = "Connected";
    conn.className = "pill ok";
    sendDraft();
    ws.send(JSON.stringify({ type: "preview", enabled: !document.hidden }));
  };
  ws.onmessage = (msg) => {
    if (typeof msg.data !== "string") {
      const url = URL.createObjectURL(msg.data);
      const img = $("#preview");
      img.onload = () => { if (previewUrl && previewUrl !== url) URL.revokeObjectURL(previewUrl); previewUrl = url; };
      img.src = url;
      return;
    }
    const data = JSON.parse(msg.data);
    if (data.type === "state") updateState(data);
    else if (data.type === "captions") addFeed(data.events);
    else if (data.type === "log") appendLog(data.lines);
    else if (data.type === "draft_error") {
      if (data.errors.length) $("#preview-note").textContent = `Preview: ${data.errors.map((e) => `${e.loc}: ${e.msg}`).join("; ")}`;
    } else if (data.type === "preview_error") $("#preview-note").textContent = `Preview: ${data.message}`;
  };
  ws.onclose = async () => {
    const conn = $("#conn");
    conn.textContent = "Offline";
    conn.className = "pill bad";
    S.ws = null;
    setTimeout(async () => {
      try { await api("GET", "/api/state"); } catch (e) { /* reloads on 401 */ }
      connect();
    }, wsRetry);
    wsRetry = Math.min(wsRetry * 2, 8000);
  };
}

document.addEventListener("visibilitychange", () => {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) {
    S.ws.send(JSON.stringify({ type: "preview", enabled: !document.hidden }));
  }
});

// ------------------------------------------------------------------ actions + shortcuts

async function control(path, body, okMessage) {
  const res = await api("POST", path, body);
  if (!res.ok) { toast(res.data?.error || `${path} failed (${res.status})`, true); return null; }
  if (res.data.state) updateState({ ...S.state, ...res.data.state });
  if (okMessage) toast(okMessage);
  return res.data;
}

async function submitManual() {
  const input = $("#manual-entry");
  const text = input.value;
  if (!text.trim()) return;
  if (await control("/api/manual-text", { text })) input.value = "";
}

const ACTIONS = {
  clear: () => control("/api/clear", {}, "Captions cleared"),
  mode_toggle: async () => {
    const r = await control("/api/mode", { mode: "toggle" });
    if (r && r.result === "manual") $("#manual-entry").focus();
    if (r) toast(r.result === "manual" ? "Manual mode" : "Auto mode");
  },
  focus_entry: () => { const i = $("#manual-entry"); i.focus(); i.select(); },
  submit: submitManual,
  pause_toggle: async () => {
    const r = await control("/api/pause", { paused: "toggle" });
    if (r) toast(r.result ? "Captions paused" : "Captions resumed");
  },
  recording_toggle: async () => {
    const r = await control("/api/recording", { recording: "toggle" });
    if (r) toast(r.result ? "Recording started" : "Recording stopped");
    refreshRecordings();
  },
  help: () => toggleHelp(),
};

// Mirrors narrator/config/shortcuts.py canonical_shortcut: modifiers in the
// order Ctrl, Alt, Shift, Meta; letters upper-cased; Shift dropped for other
// single characters (the character already encodes it: "?").
function comboFromEvent(e) {
  let key = e.key;
  if (!key || ["Control", "Shift", "Alt", "Meta", "AltGraph", "CapsLock", "Dead", "Unidentified", "Process"].includes(key)) return null;
  if (key === " ") key = "Space";
  if (key === "Esc") key = "Escape";
  let shift = e.shiftKey;
  if (key.length === 1) {
    if (/\p{L}/u.test(key)) key = key.toUpperCase();
    else shift = false;
  }
  const mods = [];
  if (e.ctrlKey) mods.push("Ctrl");
  if (e.altKey) mods.push("Alt");
  if (shift) mods.push("Shift");
  if (e.metaKey) mods.push("Meta");
  return [...mods, key].join("+");
}

function isEditable(node) {
  if (!node) return false;
  if (node.isContentEditable) return true;
  if (node.tagName === "TEXTAREA" || node.tagName === "SELECT") return true;
  if (node.tagName === "INPUT") return !["checkbox", "radio", "button", "submit", "range", "color"].includes(node.type);
  return false;
}

const TYPING_KEYS = new Set(["Enter", "Space", "Backspace", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);

function onKeyDown(e) {
  if (e.isComposing || (e.getModifierState && e.getModifierState("AltGraph"))) return; // Polish ą/ę/ł via AltGr
  const target = e.target;
  if (target && target.dataset && target.dataset.capture) return; // recording a shortcut
  const combo = comboFromEvent(e);
  if (!combo) return;
  const help = $("#help");
  if (!help.hidden && combo === "Escape") { e.preventDefault(); toggleHelp(false); return; }
  const action = Object.keys(S.shortcuts).find((a) => S.shortcuts[a] && S.shortcuts[a] === combo);
  if (!action) return;
  const key = combo.split("+").pop();
  const plain = !e.ctrlKey && !e.altKey && !e.metaKey;
  const typing = plain && (key.length === 1 || TYPING_KEYS.has(key));
  const inEntry = target === $("#manual-entry");
  if (action === "submit") {
    if (!inEntry && plain) return; // a plain key submits only from the entry
  } else if (typing && isEditable(target)) {
    return; // the operator is typing: let the key through
  }
  e.preventDefault();
  e.stopPropagation();
  ACTIONS[action]();
}

function toggleHelp(force) {
  const help = $("#help");
  const show = force === undefined ? help.hidden : force;
  if (show) {
    $("#help-table").replaceChildren(...Object.entries(ACTION_LABELS).map(([action, label]) => el("tr", {},
      el("td", { text: label }),
      el("td", {}, S.shortcuts[action] ? el("kbd", { text: S.shortcuts[action] }) : el("span", { class: "muted", text: "unbound" })))));
  }
  help.hidden = !show;
  if (show) $("#help-close").focus();
}

// ------------------------------------------------------------------ boot

async function loadDevices(refresh = false) {
  const res = await api("GET", "/api/devices" + (refresh ? "?refresh=1" : ""));
  if (res.ok) S.devices = res.data.devices;
}

async function boot() {
  document.addEventListener("keydown", onKeyDown, true);
  $("#mode-auto").addEventListener("click", () => control("/api/mode", { mode: "auto" }));
  $("#mode-manual").addEventListener("click", async () => {
    if (await control("/api/mode", { mode: "manual" })) $("#manual-entry").focus();
  });
  $("#clear-btn").addEventListener("click", ACTIONS.clear);
  $("#pause-btn").addEventListener("click", ACTIONS.pause_toggle);
  $("#rec-btn").addEventListener("click", ACTIONS.recording_toggle);
  $("#manual-form").addEventListener("submit", (e) => { e.preventDefault(); submitManual(); });
  $("#help-btn").addEventListener("click", () => toggleHelp());
  $("#help-close").addEventListener("click", () => toggleHelp(false));
  $("#help").addEventListener("click", (e) => { if (e.target.id === "help") toggleHelp(false); });
  $("#save-btn").addEventListener("click", saveConfig);
  $("#revert-btn").addEventListener("click", () => { $("#error-summary").hidden = true; loadConfig(); });
  $("#guides-toggle").addEventListener("change", updateGuides);
  $("#recheck-btn").addEventListener("click", (e) => runFix({ action: "health.recheck", label: "Check again" }, e.currentTarget));
  window.addEventListener("beforeunload", (e) => { if (isDirty()) { e.preventDefault(); e.returnValue = ""; } });

  const [schema, choices, state] = await Promise.all([
    api("GET", "/api/schema"), api("GET", "/api/choices"), api("GET", "/api/state"),
  ]);
  S.schema = schema.data;
  S.choices = choices.data;
  updateState(state.data);
  await loadDevices().catch(() => {});
  await loadConfig();
  connect();
}

boot().catch((err) => {
  console.error(err);
  toast(`The UI failed to start: ${err.message}`, true);
});
