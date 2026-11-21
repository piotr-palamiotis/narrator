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
};

const DISPLAY_SECTIONS = ["canvas", "layout", "style"];

// ------------------------------------------------------------------ schema UI

const TABS = [
  { id: "audio", title: "Audio input", sections: ["audio"] },
  { id: "asr", title: "Speech recognition", sections: ["asr"] },
  { id: "display", title: "Look & layout", sections: ["style", "layout", "canvas"] },
  { id: "ndi", title: "NDI output", sections: ["ndi"] },
  { id: "captions", title: "Word filter", sections: ["captions"] },
  { id: "remote", title: "Web & remote", sections: ["web"] },
  { id: "status", title: "Status & log", sections: ["app"] },
];

const SECTION_TITLES = {
  app: "App", audio: "Audio input", asr: "Speech recognition", canvas: "Video size", layout: "Position on screen",
  style: "Text", ndi: "NDI output", captions: "Blocked words",
  web: "Web UI & remote control", overrides: "Style overrides", background_band: "Background band",
  shadow: "Shadow", stroke: "Stroke (outline)", shortcuts: "Keyboard shortcuts", osc: "OSC input",
};

// Fields only one decoder reads are shown only while it is selected (`when`:
// every listed path must hold one of the listed values; applyConditions).
const LA = { "asr.decoder": ["local_agreement"] };
const SEG = { "asr.decoder": ["segmenter"] };

// Per-field UI hints. Keys besides label/help/when:
//   widget   - a WIDGETS entry (else picked from the schema type)
//   labels   - {enumValue: "friendly label"} for enum dropdowns / segmented
//   choices  - [[value, label], ...] for the "choice" dropdown; `custom`
//              (default true) adds an "Other…" entry with a free input
//   min/max/step/unit - slider range (the schema still validates)
//   adv      - shown under the section's "Advanced settings" fold
const SEC = (n) => [[n, `${n} s`]];
const HINTS = {
  "app.log_level": { label: "Log detail", widget: "enum", enumValues: ["DEBUG", "INFO", "WARNING", "ERROR"],
    labels: { DEBUG: "Debug (everything)", INFO: "Normal", WARNING: "Warnings and errors", ERROR: "Errors only" } },
  "audio.device": { label: "Input", widget: "device",
    help: "A microphone or sound card, or an NDI source (its audio). Rescan after plugging something in." },
  "audio.sample_rate": { label: "Sample rate", widget: "choice", adv: true,
    choices: [[16000, "16 kHz (speech recognition native)"], [44100, "44.1 kHz"], [48000, "48 kHz"]] },
  "audio.frame_ms": { label: "Audio frame length", widget: "choice", adv: true,
    choices: [[10, "10 ms"], [20, "20 ms"], [30, "30 ms (default)"]] },
  "audio.channels": { label: "Channels to listen to", widget: "choice", customLabel: "Other channels…",
    choices: [[[1], "Channel 1 (mono / left)"], [[2], "Channel 2 (right)"], [[1, 2], "Channels 1 + 2 mixed (stereo)"]],
    help: "Which inputs of the device are mixed into the speech signal. Other: a list such as 3, 4." },
  "asr.preset": { label: "Performance preset", widget: "preset" },
  "asr.model": { label: "Whisper model", widget: "model",
    help: "Bigger models are more accurate and need more GPU. Models not downloaded yet are fetched on first use." },
  "asr.language_whitelist": { label: "Languages", widget: "languages",
    help: "None = recognise any language. One = always that language. Several = the best match per sentence." },
  "asr.unit": { label: "Runs on", widget: "unit",
    help: "Auto picks the best device at start (CUDA > dedicated GPU > CPU > integrated GPU). Changing it restarts speech recognition." },
  "asr.unit_identity": { widget: "hidden" },
  "asr.vad_threshold": { label: "Speech detection sensitivity", widget: "slider", min: 0, max: 1, step: 0.05, adv: true,
    help: "Higher = needs clearer speech before captioning starts (less noise picked up)." },
  "asr.decoder": { label: "Caption style", widget: "enum",
    labels: { local_agreement: "Word by word as it is confirmed (default)", segmenter: "Whole phrases after each pause" },
    help: "Word by word re-reads the audio continuously and shows words once they are stable. Whole phrases decodes each stretch of speech once it ends (turn on partial captions for a live preview). Changing it restarts speech recognition." },
  "asr.step_ms": { label: "Decode step (ms)", when: LA, adv: true },
  "asr.agreement_passes": { label: "Agreement passes", when: LA, adv: true },
  "asr.end_silence_ms": { label: "End of sentence after silence (ms)", adv: true },
  "asr.max_utterance_ms": { label: "Max utterance (ms)", when: LA, adv: true },
  "asr.max_context_seconds": { label: "Max decode window (s)", step: 0.5, when: LA, adv: true },
  "asr.prune_history_after_silence_seconds": { label: "Forget off-screen lines after silence (s)", adv: true,
    help: "Never touches the visible lines." },
  "asr.wipe_screen_after_silence_seconds": { label: "Clear the screen after silence", widget: "choice",
    choices: [[0, "Never (keep the last captions)"], ...SEC(10), ...SEC(20), ...SEC(30), [60, "1 minute"], [120, "2 minutes"]],
    customLabel: "Other (seconds)…" },
  "asr.show_partial_captions": { label: "Show unconfirmed words",
    help: "Shows text before it is confirmed (it may still change). With whole-phrase captions this costs extra GPU time." },
  "asr.vad_preroll_ms": { label: "VAD pre-roll (ms)", when: LA, adv: true },
  "asr.prompt_carryover_chars": { label: "Prompt carry-over (chars, 0 = off)", when: LA, adv: true },
  "asr.segmentation_mode": { label: "Phrase cutting", when: SEG, adv: true,
    labels: { vad: "At speech pauses (default)", hybrid: "Fixed length, while speech is heard", fixed: "Fixed length, always (no silence detection)" } },
  "asr.partial_latency_ms": { label: "Partial preview every (ms)", when: { ...SEG, "asr.show_partial_captions": [true] }, adv: true,
    help: "Re-decode the open phrase for display after this much new audio; grows with the phrase's length and gives way when the GPU is busy." },
  "asr.segment_duration_ms": { label: "Phrase length (ms)", when: { ...SEG, "asr.segmentation_mode": ["hybrid", "fixed"] }, adv: true },
  "asr.max_segment_ms": { label: "Longest phrase (ms)", when: { ...SEG, "asr.segmentation_mode": ["vad"] }, adv: true,
    help: "Continuous speech is cut here (at a short pause once 60% full) and continues in the next phrase; nothing is lost." },
  "asr.context_sentences": { label: "Earlier captions as context", when: SEG, adv: true, widget: "choice",
    choices: [[0, "None"], [1, "1 caption (default)"], [2, "2 captions"], [3, "3 captions"]],
    help: "The last captions are given to whisper as a hint for the next phrase; reset by a screen wipe and on a language change." },
  "asr.min_mean_token_p": { label: "Min. confidence (0 = off)", widget: "slider", min: 0, max: 1, step: 0.05, adv: true },
  "asr.max_timestamp_ratio": { label: "Hallucination timestamp ratio (0 = off)", step: 0.25, adv: true },
  "asr.strip_bracketed_annotations": { label: "Hide [sound] annotations such as [music]" },
  "asr.text_filters": { label: "Find and replace (regex)", widget: "rows", adv: true,
    help: "Case-insensitive regular expressions applied before anything is captioned; an empty replacement deletes the match." },
  "asr.vocabulary": { label: "Names and terms (one per line)", widget: "lines",
    help: "Names, places and jargon the recogniser should expect." },
  "asr.vocabulary_list": { label: "Saved term list", widget: "vocablist",
    help: "Used together with the terms above. Lists live in <config dir>/vocabulary/." },
  "asr.vocabulary_snap": { label: "Correct near-miss spellings to these terms" },
  "asr.flash_attn": { label: "Flash attention (may crash some GPU drivers)", adv: true },
  "asr.dtw_word_timestamps": { label: "Exact word times (DTW, slower decodes)", adv: true,
    help: "Caption-event word times from whisper's DTW alignment; about 20% more decode time. Ignored while flash attention is on." },
  "canvas.width_px": { label: "Resolution", widget: "resolution" },
  "canvas.height_px": { widget: "hidden" },
  "canvas.background_color": { label: "Video background", widget: "rgba", adv: true,
    help: "Normally fully transparent, so the captions sit over the picture in OBS / vMix." },
  "layout.max_lines": { label: "Lines on screen", widget: "choice",
    choices: [[1, "1 line"], [2, "2 lines"], [3, "3 lines"], [4, "4 lines"]], customLabel: "More…",
    help: "The text size follows from the caption area's height and the number of lines." },
  "layout.margin_top_pct": { label: "Top margin", widget: "slider", min: 0, max: 95, step: 0.5, unit: "%" },
  "layout.margin_bottom_pct": { label: "Bottom margin", widget: "slider", min: 0, max: 95, step: 0.5, unit: "%" },
  "layout.margin_left_pct": { label: "Left margin", widget: "slider", min: 0, max: 45, step: 0.5, unit: "%" },
  "layout.margin_right_pct": { label: "Right margin", widget: "slider", min: 0, max: 45, step: 0.5, unit: "%" },
  "layout.interline_pct": { label: "Line spacing", widget: "slider", min: 0, max: 10, step: 0.1, unit: "%",
    help: "Gap between lines, in % of the picture height." },
  "layout.text_align": { label: "Text alignment", widget: "segmented", labels: { left: "Left", center: "Centre", right: "Right" } },
  "layout.vertical_align_within_box": { label: "Vertical position", widget: "segmented",
    labels: { top: "Top", center: "Middle", bottom: "Bottom" } },
  "layout.pacing": { label: "Typewriter effect",
    labels: { off: "Off (show captions at once)", char: "Letter by letter", word: "Word by word", sentence: "Sentence by sentence", smooth: "Smooth (steady letter flow)" },
    help: "Reveals captions gradually (faster when text backs up; a caption arriving alone appears at once). Smooth types at an even pace matched to how often captions arrive, so bursts flow out continuously." },
  "style.font_family": { label: "Font", widget: "font" },
  "style.font_color": { label: "Text colour", widget: "rgba" },
  "style.overrides": { widget: "effects" },
  "style.overrides.background_band": { label: "Box behind text", toggle: "enabled",
    intro: "A tinted box behind each line. The most readable on busy pictures." },
  "style.overrides.background_band.shape": { label: "Corners", widget: "segmented", labels: { pill: "Rounded", rectangle: "Square" } },
  "style.overrides.background_band.color": { label: "Box colour", widget: "rgba" },
  "style.overrides.background_band.corner_radius_pct": { label: "Roundness", widget: "slider", min: 0, max: 5, step: 0.1,
    when: { "style.overrides.background_band.shape": ["pill"] } },
  "style.overrides.background_band.padding_pct": { label: "Side padding", widget: "slider", min: 0, max: 5, step: 0.1 },
  "style.overrides.shadow": { label: "Drop shadow", toggle: "enabled",
    intro: "A soft shadow under the letters. Subtle; good together with an outline." },
  "style.overrides.shadow.offset_x_pct": { label: "Shift right", widget: "slider", min: -2, max: 2, step: 0.05 },
  "style.overrides.shadow.offset_y_pct": { label: "Shift down", widget: "slider", min: -2, max: 2, step: 0.05 },
  "style.overrides.shadow.blur_px": { label: "Softness", widget: "slider", min: 0, max: 20, step: 1, unit: "px" },
  "style.overrides.shadow.color": { label: "Shadow colour", widget: "rgba" },
  "style.overrides.stroke": { label: "Outline", toggle: "enabled",
    intro: "A thin border around every letter. Keeps text readable without covering the picture." },
  "style.overrides.stroke.width_px": { label: "Thickness", widget: "slider", min: 0, max: 12, step: 1, unit: "px" },
  "style.overrides.stroke.color": { label: "Outline colour", widget: "rgba" },
  "ndi.source_name": { label: "Source name", help: "The name OBS, vMix or Studio Monitor list this caption feed under." },
  "ndi.fps": { label: "Frame rate", widget: "choice",
    choices: [[25, "25 fps"], [29.97, "29.97 fps"], [30, "30 fps"], [50, "50 fps"], [59.94, "59.94 fps"], [60, "60 fps"]] },
  "captions.blocklist": { label: "Blocked words (one per line)", widget: "lines",
    help: "Whole words, case- and diacritics-blind; a trailing * matches every word starting with the stem (kurw*)." },
  "captions.blocklist_builtin": { label: "Also use the bundled Polish/English starter list" },
  "captions.blocklist_action": { label: "Blocked words are",
    labels: { mask: "Masked (k***a)", replace: "Replaced with the text below", drop: "Removed" } },
  "captions.blocklist_replacement": { label: "Replacement text", when: { "captions.blocklist_action": ["replace"] } },
  "web.enabled": { label: "Web UI enabled", adv: true, help: "Turning this off stops this page; re-enable in config.yaml." },
  "web.port": { label: "Port", adv: true },
  "web.preview_fps": { label: "Preview refresh rate", widget: "choice",
    choices: [[1, "1 fps"], [2, "2 fps"], [5, "5 fps (default)"], [10, "10 fps"], [15, "15 fps"], [30, "30 fps"]] },
  "web.shortcuts.clear": { label: "Clear captions", widget: "shortcut" },
  "web.shortcuts.mode_toggle": { label: "Auto / manual switch", widget: "shortcut" },
  "web.shortcuts.focus_entry": { label: "Focus manual entry", widget: "shortcut" },
  "web.shortcuts.submit": { label: "Send manual caption", widget: "shortcut",
    help: "Without Ctrl/Alt/Meta it fires only while the manual entry has focus." },
  "web.shortcuts.pause_toggle": { label: "Pause / resume captions", widget: "shortcut" },
  "web.shortcuts.help": { label: "Show this cheat sheet", widget: "shortcut" },
  "web.api_tokens": { label: "API tokens (Companion, scripts)", widget: "rows", generate: "token",
    help: "Each integration gets its own token for the control API (state, clear, mode, manual text, pause). Delete a row and Save to revoke." },
  "web.osc.enabled": { label: "OSC input enabled", help: "UDP, no authentication: keep it on this computer." },
  "web.osc.bind": { label: "Listen on", widget: "choice", when: { "web.osc.enabled": [true] },
    choices: [["127.0.0.1", "This computer only (127.0.0.1)"], ["0.0.0.0", "All networks (0.0.0.0)"]], customLabel: "Other address…" },
  "web.osc.port": { label: "OSC UDP port", when: { "web.osc.enabled": [true] } },
};

// Display order of a section's fields (unlisted fields follow, schema order).
const FIELD_ORDER = {
  asr: ["preset", "model", "unit", "language_whitelist", "decoder", "show_partial_captions",
    "wipe_screen_after_silence_seconds", "strip_bracketed_annotations", "vocabulary", "vocabulary_list", "vocabulary_snap"],
};

// One line under a section's title: what it is for.
const SECTION_INTROS = {
  audio: "Where the speech comes from.",
  asr: "How speech is turned into text. The preset sets sensible values for this computer; most of the rest can stay as it is.",
  canvas: "The size of the caption video sent over NDI.",
  layout: "Where the captions sit on the picture. Drag the margins and watch the dashed box in the preview.",
  style: "How the caption text looks.",
  ndi: "The caption video other programs receive (transparent background).",
  captions: "Words that must never appear on screen.",
  web: "This page, keyboard shortcuts and remote control (Companion, OSC).",
};

const ACTION_LABELS = {
  clear: "Clear captions", mode_toggle: "Auto / manual switch", focus_entry: "Focus manual entry",
  submit: "Send manual caption", pause_toggle: "Pause / resume captions",
  help: "Show this cheat sheet",
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

// Which "Advanced settings" folds are open (kept across re-renders).
const openFolds = new Set();

function renderObject(schema, path, container, extraWhen = null) {
  const s = resolve(schema);
  let fold = null; // this level's "Advanced settings" (created on first use)
  const advanced = () => {
    if (!fold) {
      const key = pathId(path);
      fold = el("details", { class: "advanced", open: openFolds.has(key),
        ontoggle: (e) => { if (e.target.open) openFolds.add(key); else openFolds.delete(key); } },
        el("summary", { text: "Advanced settings" }));
    }
    return fold;
  };
  const order = FIELD_ORDER[pathId(path)] || [];
  const rank = (k) => (order.includes(k) ? order.indexOf(k) : order.length);
  const entries = Object.entries(s.properties || {}).map((e, i) => [e, i])
    .sort(([[a], i], [[b], j]) => rank(a) - rank(b) || i - j).map(([e]) => e);
  for (const [key, propSchema] of entries) {
    const childPath = [...path, key];
    const id = pathId(childPath);
    const hint = HINTS[id] || {};
    const child = resolve(propSchema);
    if (hint.widget === "effects") {
      container.append(buildEffects(child, childPath));
    } else if (child.properties) {
      const fs = hint.toggle ? buildToggleCard(child, childPath, hint) : (() => {
        const box = el("fieldset", { "data-path": id }, el("legend", { text: hint.label || SECTION_TITLES[key] || humanize(key) }));
        box.append(el("div", { class: "err", id: "e-" + id }));
        renderObject(child, childPath, box, extraWhen);
        return box;
      })();
      container.append(fs);
    } else {
      const row = renderField(childPath, child, extraWhen);
      (hint.adv ? advanced() : container).append(row);
    }
  }
  if (fold) container.append(fold);
}

// A sub-section that is switched on/off by one of its boolean fields (the
// text effects): a switch in the title, the other fields only while on.
function buildToggleCard(schema, path, hint) {
  const id = pathId(path);
  const flagPath = [...path, hint.toggle];
  const on = !!getPath(S.config, flagPath);
  const card = el("fieldset", { class: "card" + (on ? " on" : ""), "data-path": id });
  const sw = el("input", { type: "checkbox", class: "switch", id: "f-" + pathId(flagPath), checked: on,
    onchange: (e) => { card.classList.toggle("on", e.target.checked); onValue(flagPath, e.target.checked); } });
  card.append(el("legend", {}, el("label", { class: "card-title" }, sw, hint.label || humanize(path[path.length - 1]))));
  if (hint.intro) card.append(el("p", { class: "help card-intro", text: hint.intro }));
  card.append(el("div", { class: "err", id: "e-" + id }), el("div", { class: "err", id: "e-" + pathId(flagPath) }));
  const body = el("div", { class: "card-body", "data-when": JSON.stringify({ [pathId(flagPath)]: [true] }) });
  const rest = { ...schema, properties: Object.fromEntries(Object.entries(schema.properties).filter(([k]) => k !== hint.toggle)) };
  renderObject(rest, path, body);
  card.append(body);
  return card;
}

// Text effects (style.overrides): quick looks + one switchable card per effect.
const LOOKS = [
  ["Plain", { background_band: false, shadow: false, stroke: false }],
  ["Outline", { background_band: false, shadow: false, stroke: true }],
  ["Outline + shadow", { background_band: false, shadow: true, stroke: true }],
  ["Box", { background_band: true, shadow: false, stroke: false }],
  ["Box + outline", { background_band: true, shadow: false, stroke: true }],
];

function buildEffects(schema, path) {
  const wrap = el("fieldset", { class: "effects", "data-path": pathId(path) },
    el("legend", { text: "Text effects" }),
    el("p", { class: "help", text: "Effects that keep the captions readable over any picture. Pick a look, then fine-tune it below — the preview shows the result before you save." }));
  const flags = () => Object.fromEntries(Object.keys(LOOKS[0][1]).map((k) => [k, !!getPath(S.config, [...path, k, "enabled"])]));
  const looks = el("div", { class: "looks", role: "group", "aria-label": "Quick looks" });
  const drawLooks = () => {
    const now = flags();
    looks.replaceChildren(el("span", { class: "small muted", text: "Quick look:" }), ...LOOKS.map(([name, set]) => el("button", {
      type: "button", text: name, "aria-pressed": String(same(now, set)),
      onclick: () => {
        for (const [k, v] of Object.entries(set)) setPath(S.config, [...path, k, "enabled"], v);
        onValue([...path, "stroke", "enabled"], set.stroke, { rebuild: true });
      } })));
  };
  drawLooks();
  wrap.append(looks, el("div", { class: "err", id: "e-" + pathId(path) }));
  const cards = el("div", { class: "cards" });
  renderObject(schema, path, cards);
  wrap.append(cards);
  wrap.addEventListener("change", drawLooks);
  return wrap;
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

function renderField(path, s, extraWhen = null) {
  const id = pathId(path);
  const hint = HINTS[id] || {};
  const widget = widgetFor(id, s, hint);
  const value = getPath(S.config, path);
  const inputId = "f-" + id;
  const control = el("div", { class: "control" });
  const wide = ["rows", "lines", "languages"].includes(widget);
  if (widget === "checkbox") return renderSwitchRow(path, id, inputId, hint, value, extraWhen);
  const row = el("div", { class: "field" + (wide ? " wide" : ""), "data-path": id });
  const label = el("label", { for: inputId, text: hint.label || humanize(path[path.length - 1]) });
  const builder = WIDGETS[widget] || WIDGETS.text;
  builder({ path, id, inputId, s, hint, value, control });
  row.append(label, control);
  if (widget === "hidden") row.hidden = true; // set by another widget (asr.unit_identity by the unit picker)
  if (hint.help) row.append(el("div", { class: "help", text: hint.help }));
  // A hidden field's errors go to its section (it has no visible row).
  if (widget !== "hidden") row.append(el("div", { class: "err", id: "e-" + id }));
  const when = { ...(extraWhen || {}), ...(hint.when || {}) };
  if (Object.keys(when).length) row.dataset.when = JSON.stringify(when);
  return row;
}

// Booleans: a switch with its label beside it (the whole row is clickable).
function renderSwitchRow(path, id, inputId, hint, value, extraWhen) {
  const row = el("div", { class: "field switch-row", "data-path": id });
  row.append(el("label", { class: "switch-label", for: inputId },
    el("input", { type: "checkbox", class: "switch", id: inputId, checked: !!value,
      onchange: (e) => onValue(path, e.target.checked) }),
    el("span", { text: hint.label || humanize(path[path.length - 1]) })));
  if (hint.help) row.append(el("div", { class: "help", text: hint.help }));
  row.append(el("div", { class: "err", id: "e-" + id }));
  const when = { ...(extraWhen || {}), ...(hint.when || {}) };
  if (Object.keys(when).length) row.dataset.when = JSON.stringify(when);
  return row;
}

// Show/hide the rows whose HINTS `when` condition depends on the current
// (unsaved) form values. A row with a validation error always stays visible.
function applyConditions() {
  for (const row of document.querySelectorAll("[data-when]")) {
    const when = JSON.parse(row.dataset.when);
    const ok = Object.entries(when).every(([p, values]) => values.includes(getPath(S.config, p.split("."))));
    const err = [...row.querySelectorAll(".err")].some((e) => e.textContent);
    row.hidden = !ok && !err;
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
  enum({ path, inputId, value, control, s, hint }) {
    const values = hint.enumValues || s.enum;
    const labels = hint.labels || {};
    const all = values.includes(value) ? values : [value, ...values];
    control.append(selectOf(inputId, all.map((v) => [v, labels[v] || v]), value, (v) => onValue(path, v)));
  },
  segmented({ path, inputId, value, control, s, hint }) {
    const labels = hint.labels || {};
    const group = el("div", { class: "segmented", role: "group", id: inputId });
    const draw = (current) => group.replaceChildren(...s.enum.map((v) => el("button", {
      type: "button", text: labels[v] || v, "aria-pressed": String(v === current),
      onclick: () => { onValue(path, v); draw(v); } })));
    draw(value);
    control.append(group);
  },
  // A dropdown of common values (hint.choices) plus "Other…" with a free input.
  choice({ path, inputId, value, control, s, hint }) {
    const options = hint.choices;
    const isList = s.type === "array";
    const numeric = s.type === "integer" || s.type === "number";
    const fmt = (v) => (Array.isArray(v) ? v.join(", ") : v ?? "");
    const parse = (raw) => {
      if (isList) return raw.split(/[,\s]+/).filter(Boolean).map((p) => (/^-?\d+$/.test(p) ? Number(p) : p));
      return numeric ? parseNumber(raw, s.type === "integer") : raw;
    };
    const index = options.findIndex(([v]) => same(v, value));
    const custom = el("input", { type: numeric ? "number" : "text", class: "custom-input", value: fmt(value),
      step: numeric ? (s.type === "integer" ? 1 : "any") : undefined, "aria-label": `${hint.label || ""} (other value)`,
      hidden: index >= 0, oninput: (e) => onValue(path, parse(e.target.value)) });
    const select = el("select", { id: inputId, onchange: (e) => {
      if (e.target.value === "other") {
        custom.hidden = false;
        custom.focus();
        onValue(path, parse(custom.value));
      } else {
        const v = clone(options[Number(e.target.value)][0]);
        custom.hidden = true;
        custom.value = fmt(v);
        onValue(path, v);
      }
    } }, options.map(([, label], i) => el("option", { value: String(i), text: label })),
      hint.custom === false ? null : el("option", { value: "other", text: hint.customLabel || "Other…" }));
    select.value = index >= 0 ? String(index) : "other";
    control.append(select, custom);
  },
  slider({ path, inputId, value, control, s, hint }) {
    const integer = s.type === "integer";
    const step = hint.step ?? (integer ? 1 : 0.1);
    const num = el("input", { type: "number", id: inputId, class: "slider-num", value: value ?? "", step,
      min: s.minimum ?? s.exclusiveMinimum, max: s.maximum ?? s.exclusiveMaximum });
    const range = el("input", { type: "range", class: "slider", min: hint.min ?? 0, max: hint.max ?? 100, step,
      value: value ?? 0, "aria-label": hint.label || "" });
    range.addEventListener("input", () => { num.value = range.value; onValue(path, parseNumber(range.value, integer)); });
    num.addEventListener("input", () => { range.value = num.value; onValue(path, parseNumber(num.value, integer)); });
    control.append(range, num);
    if (hint.unit) control.append(el("span", { class: "small muted unit", text: hint.unit }));
  },
  resolution({ inputId, control }) {
    const RES = [[1920, 1080, "1920 × 1080 — Full HD"], [1280, 720, "1280 × 720 — HD"],
      [2560, 1440, "2560 × 1440 — QHD"], [3840, 2160, "3840 × 2160 — 4K UHD"], [1080, 1920, "1080 × 1920 — vertical"]];
    const w = getPath(S.config, ["canvas", "width_px"]), h = getPath(S.config, ["canvas", "height_px"]);
    const index = RES.findIndex(([rw, rh]) => rw === w && rh === h);
    const num = (key, v) => el("input", { type: "number", class: "res-num", value: v ?? "", step: 1, min: 1,
      "aria-label": key === "width_px" ? "Width (px)" : "Height (px)",
      oninput: (e) => onValue(["canvas", key], parseNumber(e.target.value, true)) });
    const custom = el("span", { class: "res-custom", hidden: index >= 0 },
      num("width_px", w), el("span", { class: "muted", text: "×" }), num("height_px", h), el("span", { class: "small muted", text: "px" }));
    const select = el("select", { id: inputId, onchange: (e) => {
      if (e.target.value === "other") { custom.hidden = false; return; }
      const [rw, rh] = RES[Number(e.target.value)];
      setPath(S.config, ["canvas", "height_px"], rh);
      onValue(["canvas", "width_px"], rw, { rebuild: true });
    } }, RES.map(([, , label], i) => el("option", { value: String(i), text: label })), el("option", { value: "other", text: "Custom size…" }));
    select.value = index >= 0 ? String(index) : "other";
    control.append(select, custom);
  },
  model({ path, inputId, value, control }) {
    const select = el("select", { id: inputId });
    const custom = el("input", { type: "text", class: "custom-input", value: value ?? "", hidden: true,
      placeholder: "Model name or path to a ggml .bin file", "aria-label": "Model name or file",
      oninput: (e) => onValue(path, e.target.value.trim()) });
    const fill = () => {
      const current = getPath(S.config, path);
      const catalog = S.choices.catalog || [];
      const info = (m) => catalog.find((c) => c.model === m);
      const names = (S.choices.models || []).filter((m) => modelFitsUnit(m) || m === current);
      const label = (m) => {
        const c = info(m);
        const note = MODEL_NOTES[m];
        const size = c && c.size_mb ? ` · ${c.size_mb >= 1000 ? (c.size_mb / 1000).toFixed(1) + " GB" : Math.round(c.size_mb) + " MB"}` : "";
        return `${m}${note ? ` — ${note}` : ""}${size}`;
      };
      const installed = names.filter((m) => info(m)?.installed);
      const rest = names.filter((m) => !info(m)?.installed);
      select.replaceChildren(...[
        installed.length ? el("optgroup", { label: "On this computer" }, installed.map((m) => el("option", { value: m, text: label(m) }))) : null,
        rest.length ? el("optgroup", { label: "Downloaded on first use" }, rest.map((m) => el("option", { value: m, text: label(m) }))) : null,
        el("option", { value: "\u0000other", text: "Other model file…" })].filter(Boolean));
      const known = names.includes(current);
      select.value = known ? current : "\u0000other";
      custom.hidden = known;
    };
    select.addEventListener("change", (e) => {
      if (e.target.value === "\u0000other") { custom.hidden = false; custom.focus(); return; }
      custom.hidden = true;
      custom.value = e.target.value;
      onValue(path, e.target.value);
    });
    refreshModelOptions = fill;
    fill();
    control.append(select, custom);
  },
  font({ path, inputId, value, control }) {
    const fonts = S.choices.fonts || [];
    const pretty = (f) => String(f).split("/").pop().replace(/\.(ttf|otf)$/i, "")
      .replace(/[-_]/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2");
    control.append(WIDGETS_choiceOf({ path, inputId, value, options: fonts.map((f) => [f, pretty(f)]),
      customLabel: "Other font file…", placeholder: "Path to a .ttf / .otf file" }));
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
      refreshModelOptions(); // the model list offers only models that run on the chosen unit
    } });
    const note = el("div", { class: "help", id: "unit-note" });
    control.append(select, note);
    // Fill THIS select: it is not in the page yet, and a re-render (after a
    // Save) still has the old select with the same id there -- looking it up
    // by id filled the old one and left this one empty.
    refreshUnits(true, select, note);
  },
  device({ path, inputId, value, control }) {
    const deviceOptions = () => [["default", "System default input"],
      ...S.devices.filter((d) => d !== "default").map((d) => [d, d.startsWith("ndi:") ? `NDI source: ${d.slice(4)}` : d])];
    let box = WIDGETS_choiceOf({ path, inputId, value, options: deviceOptions(), customLabel: "Other (type a name)…",
      placeholder: "Device name, part of it, or ndi:<source>", missingLabel: (v) => `${v} (not found now)` });
    control.append(box, el("button", { type: "button", text: "Rescan", title: "Re-list microphones and NDI sources",
      onclick: async (e) => {
        e.target.disabled = true;
        await loadDevices(true);
        const fresh = WIDGETS_choiceOf({ path, inputId, value: getPath(S.config, path), options: deviceOptions(),
          customLabel: "Other (type a name)…", placeholder: "Device name, part of it, or ndi:<source>",
          missingLabel: (v) => `${v} (not found now)` });
        box.replaceWith(fresh);
        box = fresh;
        e.target.disabled = false;
        toast(`${S.devices.length} input(s) found`);
      } }));
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
  // Colour + opacity; the #RRGGBBAA text stays editable for exact values.
  rgba({ path, inputId, value, control }) {
    const color = el("input", { type: "color", class: "swatch", id: inputId, "aria-label": "Colour" });
    const alpha = el("input", { type: "range", min: 0, max: 255, class: "alpha", "aria-label": "Opacity" });
    const pct = el("span", { class: "small muted alpha-pct" });
    const text = el("input", { type: "text", value: value ?? "", class: "rgba-text", "aria-label": "Colour code #RRGGBBAA",
      title: "Colour code #RRGGBBAA (the last two digits are the opacity)",
      oninput: (e) => { onValue(path, e.target.value); syncPickers(); } });
    const syncPickers = () => {
      const m = /^#([0-9a-f]{6})([0-9a-f]{2})$/i.exec(text.value);
      if (m) { color.value = "#" + m[1].toLowerCase(); alpha.value = parseInt(m[2], 16); }
      pct.textContent = `${Math.round((Number(alpha.value) / 255) * 100)}% opaque`;
      alpha.style.setProperty("--swatch", color.value);
    };
    const fromPickers = () => {
      const a = Number(alpha.value).toString(16).padStart(2, "0").toUpperCase();
      text.value = color.value.toUpperCase() + a;
      syncPickers();
      onValue(path, text.value);
    };
    color.addEventListener("input", fromPickers);
    alpha.addEventListener("input", fromPickers);
    syncPickers();
    control.append(color, el("span", { class: "alpha-wrap" }, alpha, pct), text);
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
      note.textContent = p ? p.note : "No preset: the model is chosen below, decode timing under Advanced settings.";
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
    const editor = el("div", { class: "vocab-editor", hidden: true });
    const lists = S.choices.vocabulary_lists || [];
    control.append(
      WIDGETS_choiceOf({ path, inputId, value, options: [["", "None"], ...lists.map((v) => [v, v])],
        customLabel: "New list…", placeholder: "Name for the new list" }),
      el("button", { type: "button", text: "Edit list…",
        onclick: () => openVocabEditor(String(getPath(S.config, path) || "").trim(), editor) }),
      editor,
    );
  },
};

// A string dropdown with an "Other…" free-text entry; the current value is
// kept as an option when it is not in the list (missingLabel names it).
function WIDGETS_choiceOf({ path, inputId, value, options, customLabel, placeholder, missingLabel }) {
  const wrap = el("span", { class: "choice" });
  const opts = [...options];
  if (value && !opts.some(([v]) => v === value) && missingLabel) opts.push([value, missingLabel(value)]);
  const known = opts.some(([v]) => v === value);
  const custom = el("input", { type: "text", class: "custom-input", value: known ? "" : value ?? "", placeholder,
    hidden: known, "aria-label": placeholder, oninput: (e) => onValue(path, e.target.value.trim()) });
  const select = el("select", { id: inputId, onchange: (e) => {
    if (e.target.value === "\u0000other") { custom.hidden = false; custom.focus(); onValue(path, custom.value.trim()); return; }
    custom.hidden = true;
    onValue(path, e.target.value);
  } }, opts.map(([v, label]) => el("option", { value: v, text: label })), el("option", { value: "\u0000other", text: customLabel }));
  select.value = known ? value : "\u0000other";
  wrap.append(select, custom);
  return wrap;
}

// One-line hints for well-known whisper models in the model dropdown.
const MODEL_NOTES = {
  tiny: "fastest, least accurate", base: "very fast", small: "fast, good for a CPU",
  medium: "balanced", "large-v1": "accurate", "large-v2": "accurate", "large-v3": "most accurate, needs a good GPU",
  "large-v3-turbo": "nearly as accurate, much faster",
};
let refreshModelOptions = () => {};

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

// Short device name for the picker: no (R)/(TM), CPU clock or chip codename.
function shortDeviceName(name) {
  return String(name || "").replace(/\((R|TM)\)/gi, "").replace(/\s*@.*$/, "").replace(/\s+CPU\b/i, "")
    .replace(/\s*\([A-Z0-9 ]+\)\s*$/, "").replace(/\s+/g, " ").trim();
}

// "<backend> — <device>", e.g. "Vulkan — NVIDIA GeForce RTX 3070". The index
// is added only when one backend lists two devices with the same name.
function unitOptionLabel(u, units = []) {
  const name = shortDeviceName(u.name);
  const twin = units.some((o) => o.id !== u.id && o.kind === u.kind && shortDeviceName(o.name) === name);
  const tags = [u.type === "integrated" ? "integrated" : "", u.usable ? "" : "not available"].filter(Boolean);
  return `${u.backend} — ${name}${twin ? ` #${u.index}` : ""}${tags.length ? ` (${tags.join(", ")})` : ""}`;
}

// Re-render the unit picker's options (only when the unit list, the model
// or the selection changed, so an open dropdown is not replaced).
let unitsKey = null;
function refreshUnits(force = false, select = document.getElementById("f-asr.unit"),
                      note = document.getElementById("unit-note")) {
  if (!select || !S.config) return;
  const units = S.state.units || [];
  const value = getPath(S.config, ["asr", "unit"]) || "auto";
  const kinds = catalogUnits(getPath(S.config, ["asr", "model"]));
  const key = JSON.stringify([units.map((u) => [u.id, u.usable, u.recommended]), value, kinds]);
  if (force || key !== unitsKey) {
    unitsKey = key;
    const rec = units.find((u) => u.recommended);
    const options = [el("option", { value: "auto", text: `Auto${rec ? ` (${unitOptionLabel(rec, units)})` : ""}` })];
    for (const u of units) {
      const fits = kinds.includes(u.kind);
      options.push(el("option", { value: u.id, text: unitOptionLabel(u, units) + (fits ? "" : " (not for this model)"),
        disabled: (!u.usable || !fits) && u.id !== value, title: u.reason || (fits ? "" : `the model runs on ${kinds.join(", ")}`) }));
    }
    if (value !== "auto" && !units.some((u) => u.id === value)) {
      options.push(el("option", { value, text: units.length ? `${value} — not on this machine` : value }));
    }
    select.replaceChildren(...options);
    select.value = value;
  }
  if (note) {
    // Only when something needs attention; the reason of a greyed-out
    // option is in its tooltip.
    const chosen = units.find((u) => u.id === value);
    const parts = [];
    if (value === "auto" && units.length && !S.state.unit_resolved) parts.push("Auto: no usable device found.");
    if (chosen && !chosen.usable) parts.push(`Not available: ${chosen.reason}.`);
    if (!units.length) parts.push("Listing devices…");
    note.textContent = parts.join(" ");
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
    onclick: () => { S.activeTab = t.id; renderTabs(); if (t.id === "display") autoSample(); },
  })));
  const tab = tabs.find((t) => t.id === S.activeTab) || tabs[0];
  const panel = el("div", { role: "tabpanel", "aria-labelledby": "tab-" + tab.id });
  if (tab.id === "status") panel.append(buildStatus());
  for (const section of tab.sections) {
    const schema = S.schema.properties[section];
    if (!schema) continue;
    const fs = el("fieldset", { "data-path": section }, el("legend", { text: SECTION_TITLES[section] || humanize(section) }));
    if (SECTION_INTROS[section]) fs.append(el("p", { class: "help section-intro", text: SECTION_INTROS[section] }));
    fs.append(el("div", { class: "err", id: "e-" + section }));
    renderObject(schema, [section], fs);
    panel.append(fs);
  }
  if (tab.id === "remote") panel.append(buildPasswordBox(), buildRemoteHelp());
  panels.replaceChildren(panel);
  panels.scrollTop = scroll;
  showErrors();
  applyConditions();
  if (tab.id === "status") renderLog();
}

// Set / change the login password (POST /api/password). Not part of the
// settings form: the hash lives in its own file, never in config.yaml. With
// no password yet the page is open (loopback bind) and no current password
// is asked for.
function buildPasswordBox() {
  const field = (id, label, autocomplete) => el("div", { class: "field" },
    el("label", { for: id, text: label }),
    el("div", { class: "control" }, el("input", { type: "password", id, autocomplete })));
  const msg = el("p", { class: "small", id: "pw-msg" });
  const note = el("p", { class: "small err", id: "pw-note", hidden: true,
    text: "No password is set: anyone on this computer can open this page without logging in. Set one to require a login." });
  const current = field("pw-current", "Current password", "current-password");
  const legend = el("legend", { text: "Login password" });
  const button = el("button", { type: "button", class: "primary", text: "Change password" });
  let isSet = true;
  const refresh = async () => {
    try {
      const info = await (await fetch("/api/login-info", { credentials: "same-origin" })).json();
      isSet = !!info.password_set;
    } catch (e) { isSet = true; }
    current.hidden = !isSet;
    note.hidden = isSet;
    legend.textContent = isSet ? "Login password" : "Set a login password";
    button.textContent = isSet ? "Change password" : "Set password";
  };
  button.addEventListener("click", async () => {
    const next = $("#pw-new").value;
    if (next !== $("#pw-confirm").value) { msg.className = "small err"; msg.textContent = "The new passwords do not match."; return; }
    button.disabled = true;
    try {
      const res = await api("POST", "/api/password", { current: isSet ? $("#pw-current").value : "", new: next });
      if (res.ok) {
        for (const id of ["#pw-current", "#pw-new", "#pw-confirm"]) $(id).value = "";
        msg.className = "small";
        msg.textContent = isSet ? "Password changed. Other browsers were logged out."
          : "Password set. This browser stays logged in; others need the password from now on.";
        await refresh();
      } else {
        msg.className = "small err"; msg.textContent = res.data?.error || `Not changed (${res.status})`;
      }
    } finally { button.disabled = false; }
  });
  refresh();
  return el("fieldset", { id: "password-box" }, legend, note, current,
    field("pw-new", "New password (8+ characters)", "new-password"),
    field("pw-confirm", "New password again", "new-password"), button, msg);
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
      "",
      "OSC: /narrator/clear, /narrator/mode [s], /narrator/manual-text s,",
      "     /narrator/pause [i|T|F]",
    ].join("\n") }),
    el("p", { class: "small muted", text: "Bitfocus Companion: see the README section \"Web UI and remote control\" for a ready button set." }));
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

// A sample caption in this browser's preview only (never on air), so the
// look can be set up while nobody is speaking.
const SAMPLE_TEXT = "Przykładowy napis — tak wyglądają napisy.";
let sampleAuto = false;
function sendSample() {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) return;
  const on = $("#sample-toggle").checked;
  S.ws.send(JSON.stringify({ type: "sample", text: on ? SAMPLE_TEXT : "" }));
  updatePreviewNote();
}
function autoSample() {
  if (sampleAuto) return;
  sampleAuto = true;
  if (!$("#sample-toggle").checked) { $("#sample-toggle").checked = true; sendSample(); }
}
let draftNote = "";
function updatePreviewNote() {
  const parts = [];
  if ($("#sample-toggle").checked) parts.push("Showing sample text (not on air)");
  if (draftNote) parts.push(draftNote);
  $("#preview-note").textContent = parts.join(" · ");
}

function displayOf(cfg) { return Object.fromEntries(DISPLAY_SECTIONS.map((k) => [k, cfg[k]])); }

function sendDraft() {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN || !S.config) return;
  const draft = displayOf(S.config);
  const unsaved = !same(draft, displayOf(S.saved));
  S.ws.send(JSON.stringify({ type: "draft", display: unsaved ? draft : null }));
  draftNote = unsaved ? "Previewing unsaved look settings — Save to put them on air" : "";
  updatePreviewNote();
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
  $("#manual-hint").textContent = manual
    ? "Manual mode: ASR output is not put on air."
    : (st.captions_paused ? "Captions paused: ASR keeps listening, nothing new goes on air." : "");
  const level = Math.max(0, Math.min(1, st.mic_level || 0));
  const bar = $("#meter-bar");
  bar.style.width = `${Math.pow(level, 1 / 3) * 100}%`;
  bar.classList.toggle("clip", level >= 0.98);
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
const SUBSYSTEM_TITLES = { asr: "ASR", audio: "Audio", ndi: "NDI", web: "Web", setup: "Setup" };
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
    sendSample();
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
  focus_entry: () => {
    if (!document.body.classList.contains("mode-manual")) return; // the entry exists in manual mode only
    const i = $("#manual-entry"); i.focus(); i.select();
  },
  submit: submitManual,
  pause_toggle: async () => {
    const r = await control("/api/pause", { paused: "toggle" });
    if (r) toast(r.result ? "Captions paused" : "Captions resumed");
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
  $("#manual-form").addEventListener("submit", (e) => { e.preventDefault(); submitManual(); });
  $("#help-btn").addEventListener("click", () => toggleHelp());
  $("#help-close").addEventListener("click", () => toggleHelp(false));
  $("#help").addEventListener("click", (e) => { if (e.target.id === "help") toggleHelp(false); });
  $("#save-btn").addEventListener("click", saveConfig);
  $("#revert-btn").addEventListener("click", () => { $("#error-summary").hidden = true; loadConfig(); });
  $("#guides-toggle").addEventListener("change", updateGuides);
  $("#sample-toggle").addEventListener("change", sendSample);
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
