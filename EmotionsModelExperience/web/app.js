/**
 * Emotions Model Experience — Frontend
 *
 * Manages:
 *  - WebSocket connection to the Python orchestrator
 *  - Registration screen (BLE scan, folder browse, form submit)
 *  - Participant display state machine (screens)
 *  - Countdown ring and progress bars
 *  - Webcam mirror (getUserMedia)
 *  - Experimenter overlay (F1 toggle)
 */

"use strict";

// ── Config ──────────────────────────────────────────────────────────────────
const WS_URL             = `ws://${location.host}/ws`;
const WS_RETRY_MS        = 2000;
const RING_CIRCUMFERENCE = 339.3;  // 2π × 54

// ── DOM refs ────────────────────────────────────────────────────────────────
const screens = {
  registration: document.getElementById("screen-registration"),
  waiting:      document.getElementById("screen-waiting"),
  consent:      document.getElementById("screen-consent"),
  baseline:     document.getElementById("screen-baseline"),
  fixation:     document.getElementById("screen-fixation"),
  stimulus:     document.getElementById("screen-stimulus"),
  relax:        document.getElementById("screen-relax"),
  rest:         document.getElementById("screen-rest"),
  done:         document.getElementById("screen-done"),
};

const els = {
  // Registration
  wsDot:          document.getElementById("ws-dot"),
  inpPid:         document.getElementById("inp-pid"),
  inpSession:     document.getElementById("inp-session"),
  inpExperimenter:document.getElementById("inp-experimenter"),
  inpDataDir:     document.getElementById("inp-data-dir"),
  btnBrowse:      document.getElementById("btn-browse"),
  btnScan:        document.getElementById("btn-scan"),
  scanSpinner:    document.getElementById("scan-spinner"),
  bleSection:     document.getElementById("ble-section"),
  deviceSelectWrap: document.getElementById("device-select-wrap"),
  selDevice:      document.getElementById("sel-device"),
  scanStatus:     document.getElementById("scan-status"),
  connectStatus:  document.getElementById("connect-status"),
  btnRegister:    document.getElementById("btn-register"),
  // Waiting
  waitingSessionInfo: document.getElementById("waiting-session-info"),
  // Participant display
  mirrorContainer: document.getElementById("mirror-container"),
  mirrorVideo:     document.getElementById("mirror-video"),
  baselineBar:     document.getElementById("baseline-bar"),
  baselineLabel:   document.getElementById("baseline-label"),
  // Side cues
  sideCueLeft:  document.getElementById("side-cue-left"),
  sideCueRight: document.getElementById("side-cue-right"),
  // Stimulus — normal layout
  stimNormal:      document.getElementById("stim-normal"),
  stimNameA:       document.getElementById("stim-emotion-name-a"),
  stimNameEnA:     document.getElementById("stim-emotion-name-en-a"),
  stimInstrFrA:    document.getElementById("stim-instruction-fr-a"),
  stimInstrEnA:    document.getElementById("stim-instruction-en-a"),
  // Stimulus — mirror layout
  stimMirrorOverlay: document.getElementById("stim-mirror-overlay"),
  stimNameB:         document.getElementById("stim-emotion-name-b"),
  stimNameEnB:       document.getElementById("stim-emotion-name-en-b"),
  stimInstrFrB:      document.getElementById("stim-instruction-fr-b"),
  stimInstrEnB:      document.getElementById("stim-instruction-en-b"),
  ringArcM:          document.getElementById("ring-arc-m"),
  countdownNumberM:  document.getElementById("countdown-number-m"),
  // Shared countdown ring (normal layout)
  ringArc:         document.getElementById("ring-arc"),
  countdownNumber: document.getElementById("countdown-number"),
  relayBar:        document.getElementById("relax-bar"),
  restBar:         document.getElementById("rest-bar"),
  restCountdown:   document.getElementById("rest-countdown"),
  // Overlay
  expOverlay:      document.getElementById("exp-overlay"),
  expWsStatus:     document.getElementById("exp-ws-status"),
  expPid:          document.getElementById("exp-pid"),
  expBlockTrial:   document.getElementById("exp-block-trial"),
  expElapsed:      document.getElementById("exp-elapsed"),
  expState:        document.getElementById("exp-state"),
  expMirror:       document.getElementById("exp-mirror"),
  expTrialList:    document.getElementById("exp-trial-list"),
  expNotes:        document.getElementById("exp-notes"),
  btnStart:        document.getElementById("btn-start"),
  btnPause:        document.getElementById("btn-pause"),
  btnResume:       document.getElementById("btn-resume"),
  btnAbort:        document.getElementById("btn-abort"),
  btnExpressions:  document.getElementById("btn-expressions"),
  btnLang:         document.getElementById("btn-lang"),
  lightbox:        document.getElementById("lightbox"),
  lightboxBackdrop:document.getElementById("lightbox-backdrop"),
  lightboxClose:   document.getElementById("lightbox-close"),
  // Consent
  consentFirst:    document.getElementById("inp-consent-first"),
  consentLast:     document.getElementById("inp-consent-last"),
  consentEmail:    document.getElementById("inp-consent-email"),
  consentError:    document.getElementById("consent-field-error"),
  btnConsentAccept:document.getElementById("btn-consent-accept"),
  btnConsentRefuse:document.getElementById("btn-consent-refuse"),
};

// ── State ────────────────────────────────────────────────────────────────────
let ws            = null;
let wsConnected   = false;
let isDryRun      = false;
let animRAF       = null;
let streamHandle  = null;   // active MediaStream (kept alive for full session)
let mediaRecorder = null;   // MediaRecorder instance during session
let currentBlock  = { number: 0, mirror: false, trials: [], currentTrial: 0 };
let lang          = "fr";   // "fr" | "en" — participant display language
let baselineWhich = null;   // "PRE" | "POST" — remembered to re-label on lang switch

// ── Screen transitions ───────────────────────────────────────────────────────

function showScreen(name) {
  if (animRAF) { cancelAnimationFrame(animRAF); animRAF = null; }
  Object.entries(screens).forEach(([k, el]) => {
    el.classList.toggle("active", k === name);
  });
}

// ── Progress bar helpers ─────────────────────────────────────────────────────

function animateProgressBar(barEl, duration) {
  const start = performance.now();
  function tick(now) {
    const elapsed = (now - start) / 1000;
    const frac = Math.max(0, 1 - elapsed / duration);
    barEl.style.transition = "none";
    barEl.style.transform  = `scaleX(${frac})`;
    if (frac > 0) animRAF = requestAnimationFrame(tick);
  }
  animRAF = requestAnimationFrame(tick);
}

// ── Countdown ring ───────────────────────────────────────────────────────────

function animateRing(duration, mirrorMode) {
  const arcEl  = mirrorMode ? els.ringArcM        : els.ringArc;
  const numEl  = mirrorMode ? els.countdownNumberM : els.countdownNumber;
  const start  = performance.now();

  // Reset whichever ring is NOT in use
  const idleArc = mirrorMode ? els.ringArc        : els.ringArcM;
  const idleNum = mirrorMode ? els.countdownNumber : els.countdownNumberM;
  if (idleArc) idleArc.setAttribute("stroke-dashoffset", "0");
  if (idleNum) idleNum.textContent = "";

  function tick(now) {
    const elapsed = (now - start) / 1000;
    const frac    = Math.max(0, 1 - elapsed / duration);
    arcEl.setAttribute("stroke-dashoffset", (RING_CIRCUMFERENCE * (1 - frac)).toFixed(2));
    const remaining = Math.ceil(duration - elapsed);
    numEl.textContent = remaining > 0 ? remaining : "";
    if (frac > 0) animRAF = requestAnimationFrame(tick);
  }
  animRAF = requestAnimationFrame(tick);
}

// ── Rest countdown ───────────────────────────────────────────────────────────

function animateRest(duration) {
  const start = performance.now();
  function tick(now) {
    const elapsed   = (now - start) / 1000;
    const remaining = Math.max(0, Math.ceil(duration - elapsed));
    const frac      = Math.max(0, 1 - elapsed / duration);
    els.restCountdown.textContent = remaining + " s";
    els.restBar.style.transition  = "none";
    els.restBar.style.transform   = `scaleX(${frac})`;
    if (remaining > 0) animRAF = requestAnimationFrame(tick);
  }
  animRAF = requestAnimationFrame(tick);
}

// ── Webcam / recording ───────────────────────────────────────────────────────

/**
 * Acquire the camera stream once and keep it alive for the full session.
 * The stream is reused for both the mirror display and the MediaRecorder.
 */
async function initCamera() {
  if (streamHandle) return true;
  try {
    streamHandle = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    els.mirrorVideo.srcObject = streamHandle;
    return true;
  } catch (e) {
    console.warn("[Camera] getUserMedia failed:", e);
    return false;
  }
}

/**
 * Show the square area in the center of the screen.
 * withCamera=true  → live webcam feed (mirror blocks)
 * withCamera=false → black rectangle (non-mirror blocks)
 * The square is always the same size and position; only the content differs.
 */
function showSquare(withCamera) {
  els.mirrorContainer.classList.add("visible");
  if (withCamera) {
    if (streamHandle) els.mirrorVideo.srcObject = streamHandle;
  } else {
    els.mirrorVideo.srcObject = null;   // black square
  }
}

/** Hide the square entirely (baseline, rest, registration, done). */
function hideSquare() {
  els.mirrorContainer.classList.remove("visible");
  els.mirrorVideo.srcObject = null;
  setSideCue(null);
}

/**
 * Show the '<' or '>' directional cue beside the square.
 * side: "LEFT" | "RIGHT" | null (null = hide both)
 */
function setSideCue(side) {
  els.sideCueLeft.classList.toggle("hidden",  side !== "LEFT");
  els.sideCueRight.classList.toggle("hidden", side !== "RIGHT");
}

/**
 * Start recording.  Must be called after initCamera() succeeds.
 * Sends binary WebM chunks over the WebSocket as they become available.
 * Also sends a video_start text message first with the epoch timestamp
 * so the server can write a synchronisation marker.
 */
async function startVideoRecording() {
  if (!streamHandle) {
    const ok = await initCamera();
    if (!ok) { console.warn("[Recording] No camera — skipping video."); return; }
  }
  if (mediaRecorder) return;

  // Pick best available codec
  const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
    .find(m => MediaRecorder.isTypeSupported(m)) || "";

  try {
    mediaRecorder = new MediaRecorder(streamHandle, mimeType ? { mimeType } : {});
  } catch (e) {
    console.warn("[Recording] MediaRecorder init failed:", e);
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0 && ws && wsConnected) {
      ws.send(e.data);   // binary frame — server appends to .webm file
    }
  };

  // Announce start timestamp BEFORE the first chunk arrives
  const startTs = Date.now() / 1000;
  send({ type: "video_start", timestamp: startTs });

  mediaRecorder.start(500);  // emit a chunk every 500 ms
  console.log("[Recording] Started —", mimeType || "default codec");
}

/** Stop recording and release the camera. */
function stopVideoRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder = null;
    send({ type: "video_stop" });
  }
  // Release camera stream
  if (streamHandle) {
    streamHandle.getTracks().forEach(t => t.stop());
    streamHandle = null;
    els.mirrorVideo.srcObject = null;
  }
  els.mirrorContainer.classList.remove("visible");
  console.log("[Recording] Stopped.");
}

// ── WebSocket ────────────────────────────────────────────────────────────────

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsConnected = true;
    setWsStatus(true);
    console.log("[WS] connected");
  };

  ws.onclose = ws.onerror = () => {
    wsConnected = false;
    setWsStatus(false);
    console.log("[WS] disconnected — retrying in", WS_RETRY_MS, "ms");
    setTimeout(connect, WS_RETRY_MS);
  };

  ws.onmessage = ({ data }) => {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    handleMessage(msg);
  };
}

function send(obj) {
  if (ws && wsConnected) ws.send(JSON.stringify(obj));
}

function setWsStatus(ok) {
  els.wsDot.className        = "ws-indicator " + (ok ? "connected" : "disconnected");
  els.expWsStatus.textContent = ok ? "connecté" : "déconnecté";
  els.expWsStatus.className   = "badge " + (ok ? "badge-ok" : "badge-err");
}

// ── Message handler ──────────────────────────────────────────────────────────

function handleMessage(msg) {
  switch (msg.type) {

    // ── Server handshake ───────────────────────────────────────────────────
    case "server_info":
      isDryRun = msg.dry_run;
      if (isDryRun) {
        // Hide BLE section; submit is always available once PID is filled
        els.bleSection.style.display = "none";
        checkRegisterReady();
      }
      break;

    // ── Registration flow ──────────────────────────────────────────────────
    case "scan_started":
      setScanStatus("Recherche en cours…");
      els.scanSpinner.classList.remove("hidden");
      els.btnScan.disabled = true;
      break;

    case "scan_results": {
      els.scanSpinner.classList.add("hidden");
      els.btnScan.disabled = false;

      const devices = msg.devices || [];
      els.selDevice.innerHTML = '<option value="">— Sélectionner un appareil —</option>';
      devices.forEach(d => {
        const opt = document.createElement("option");
        opt.value       = d.address;
        opt.textContent = `${d.name}  [${d.address}]`;
        // Pre-select Hermes if found
        if (d.name.toLowerCase().includes("hermes")) opt.selected = true;
        els.selDevice.appendChild(opt);
      });

      if (devices.length === 0) {
        setScanStatus("Aucun appareil Hermes trouvé. Vérifiez le casque et réessayez.", "err");
      } else {
        setScanStatus(`${devices.length} appareil(s) Hermes trouvé(s).`, "ok");
        els.deviceSelectWrap.classList.remove("hidden");
      }
      checkRegisterReady();
      break;
    }

    case "scan_error":
      els.scanSpinner.classList.add("hidden");
      els.btnScan.disabled = false;
      setScanStatus("Erreur Bluetooth : " + msg.msg, "err");
      break;

    case "scan_not_available":
      els.bleSection.style.display = "none";
      checkRegisterReady();
      break;

    case "connecting":
      setConnectStatus("Connexion à " + msg.mac + "…");
      els.btnRegister.disabled = true;
      break;

    case "connected":
      setConnectStatus("Connecté ✓", "ok");
      break;

    case "connect_error":
      setConnectStatus("Erreur : " + msg.msg, "err");
      els.btnRegister.disabled = false;
      break;

    case "register_error":
      setConnectStatus("Erreur : " + msg.msg, "err");
      els.btnRegister.disabled = false;
      break;

    case "registered":
      els.waitingSessionInfo.textContent =
        `${msg.pid}  ·  Session ${String(msg.session).padStart(2, "0")}`;
      showScreen("waiting");
      // Open experimenter overlay automatically after registration
      els.expOverlay.classList.remove("hidden");
      setButtons({ start: true, pause: false, resume: false, abort: false });
      break;

    // ── Status ────────────────────────────────────────────────────────────
    case "status":
      updateOverlayStatus(msg);
      break;

    // ── Session flow ───────────────────────────────────────────────────────
    case "session_start":
      hideSquare();
      setButtons({ start: false, pause: true, resume: false, abort: true });
      // Acquire camera + start recording (user gesture still live from Démarrer click)
      startVideoRecording();
      break;

    case "baseline":
      baselineWhich = msg.which;
      updateBaselineLabel();
      els.baselineBar.style.transform = "scaleX(1)";
      showScreen("baseline");
      animateProgressBar(els.baselineBar, msg.duration);
      break;

    case "block_start":
      currentBlock = {
        number:       msg.block,
        mirror:       msg.mirror,
        trials:       msg.trials || [],
        currentTrial: 0,
      };
      showSquare(msg.mirror);
      els.expMirror.textContent = msg.mirror ? "ON" : "OFF";
      renderTrialList();
      break;

    case "block_end":
      hideSquare();
      break;

    case "fixation":
      // Used only for baseline periods; not sent during trials any more.
      hideSquare();
      showScreen("fixation");
      break;

    case "stimulus": {
      // Always use the overlay layout (text top, square center)
      els.stimNormal.classList.add("hidden");
      els.stimMirrorOverlay.classList.remove("hidden");

      // Populate the overlay variant only
      els.stimNameB.textContent    = msg.label_fr;
      els.stimNameEnB.textContent  = msg.label_en;
      els.stimInstrFrB.textContent = msg.instruction_fr;
      els.stimInstrEnB.textContent = msg.instruction_en;

      // Reset mirror ring
      if (els.ringArcM) els.ringArcM.setAttribute("stroke-dashoffset", "0");

      showScreen("stimulus");
      animateRing(msg.duration, true);   // always use mirror-layout ring
      showSquare(msg.mirror);            // camera feed or black square
      setSideCue(msg.side ?? null);      // '<' or '>' for contempt, null otherwise

      currentBlock.currentTrial = msg.trial;
      renderTrialList();

      send({
        type:      "ack",
        emotion:   msg.emotion,
        block:     msg.block,
        trial:     msg.trial,
        timestamp: Date.now() / 1000,
      });
      break;
    }

    case "relax": {
      // Always use the overlay layout; square is black or camera depending on block
      document.getElementById("relax-normal").classList.add("hidden");
      document.getElementById("relax-mirror-overlay").classList.remove("hidden");
      const relaxBarEl = document.getElementById("relax-bar-m");
      relaxBarEl.style.transform = "scaleX(1)";
      showScreen("relax");
      showSquare(currentBlock.mirror);
      setSideCue(null);   // no directional cue during relax
      animateProgressBar(relaxBarEl, msg.duration);
      break;
    }

    case "rest":
      hideSquare();
      showScreen("rest");
      animateRest(msg.duration);
      break;

    case "session_end":
      hideSquare();
      showScreen("done");
      stopVideoRecording();   // finalises the .webm file and releases camera
      setButtons({ start: false, pause: false, resume: false, abort: false });
      flushNotes();
      break;

    case "error":
      console.error("[WS Server error]", msg.msg);
      break;
  }
}

// ── Registration helpers ─────────────────────────────────────────────────────

function setScanStatus(text, kind = "") {
  els.scanStatus.textContent = text;
  els.scanStatus.className   = "scan-status" + (kind ? " " + kind : "");
}

function setConnectStatus(text, kind = "") {
  els.connectStatus.textContent = text;
  els.connectStatus.className   = "connect-status" + (kind ? " " + kind : "");
  els.connectStatus.classList.remove("hidden");
}

function checkRegisterReady() {
  const hasPid    = els.inpPid.value.trim().length > 0;
  const hasDevice = isDryRun || (els.selDevice.value !== "");
  els.btnRegister.disabled = !(hasPid && hasDevice);
}

// ── Overlay helpers ───────────────────────────────────────────────────────────

function updateOverlayStatus(s) {
  els.expPid.textContent        = s.pid ? `${s.pid} / S${String(s.session ?? "?").padStart(2, "0")}` : "—";
  els.expBlockTrial.textContent = s.block ? `B${s.block} / T${s.trial ?? "?"}` : "—";
  els.expElapsed.textContent    = s.elapsed_s != null ? formatDuration(s.elapsed_s) : "0 s";
  els.expState.textContent      = s.state ?? "—";
  if (s.paused) {
    setButtons({ start: false, pause: false, resume: true, abort: true });
  }
}

function renderTrialList() {
  els.expTrialList.innerHTML = "";
  currentBlock.trials.forEach((emotion, i) => {
    const chip = document.createElement("div");
    chip.className   = "trial-chip";
    chip.textContent = emotion.replace(/_/g, " ");
    const trialNum = i + 1;
    if (trialNum <  currentBlock.currentTrial) chip.classList.add("done");
    if (trialNum === currentBlock.currentTrial) chip.classList.add("current");
    els.expTrialList.appendChild(chip);
  });
}

function setButtons({ start, pause, resume, abort }) {
  els.btnStart.disabled  = !start;
  els.btnPause.disabled  = !pause;
  els.btnAbort.disabled  = !abort;
  els.btnResume.disabled = !resume;
  els.btnPause.style.display  = resume ? "none" : "";
  els.btnResume.style.display = resume ? ""     : "none";
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m} min ${String(s).padStart(2, "0")} s` : `${s} s`;
}

function flushNotes() {
  const notes = els.expNotes.value.trim();
  if (notes) send({ type: "notes", notes });
}

// ── Expressions lightbox ──────────────────────────────────────────────────────

function openLightbox() {
  els.lightbox.classList.remove("hidden");
}

function closeLightbox() {
  els.lightbox.classList.add("hidden");
}

// ── Language toggle ───────────────────────────────────────────────────────────

const BASELINE_LABELS = {
  fr: { PRE: "Ligne de base — début de session", POST: "Ligne de base — fin de session" },
  en: { PRE: "Baseline — session start",         POST: "Baseline — session end" },
};

function updateBaselineLabel() {
  if (!baselineWhich) return;
  els.baselineLabel.textContent = BASELINE_LABELS[lang][baselineWhich] ?? "";
}

function setLang(newLang) {
  lang = newLang;
  if (lang === "en") {
    document.documentElement.classList.add("lang-en");
    els.btnLang.textContent = "EN";
    els.btnLang.title = "Switch to French / Passer en français";
  } else {
    document.documentElement.classList.remove("lang-en");
    els.btnLang.textContent = "FR";
    els.btnLang.title = "Switch to English";
  }
  updateBaselineLabel();
}

// ── Consent screen ────────────────────────────────────────────────────────────

function showConsentScreen() {
  // Clear previous values and errors
  els.consentFirst.value  = "";
  els.consentLast.value   = "";
  els.consentEmail.value  = "";
  els.consentError.textContent = "";
  els.consentError.classList.add("hidden");
  // Hide experimenter overlay, show consent to participant
  els.expOverlay.classList.add("hidden");
  showScreen("consent");
}

function submitConsent() {
  const first = els.consentFirst.value.trim();
  const last  = els.consentLast.value.trim();
  const email = els.consentEmail.value.trim();
  const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  if (!first || !last || !email) {
    const msg = lang === "fr"
      ? "Veuillez remplir tous les champs."
      : "Please fill in all fields.";
    els.consentError.textContent = msg;
    els.consentError.classList.remove("hidden");
    return;
  }
  if (!emailOk) {
    const msg = lang === "fr"
      ? "Adresse courriel invalide."
      : "Invalid email address.";
    els.consentError.textContent = msg;
    els.consentError.classList.remove("hidden");
    return;
  }

  send({
    type:       "consent",
    first_name: first,
    last_name:  last,
    email:      email,
    lang:       lang,
    timestamp:  Date.now() / 1000,
  });
  send({ type: "start" });
  setButtons({ start: false, pause: true, resume: false, abort: true });
}

function refuseConsent() {
  showScreen("waiting");
  els.expOverlay.classList.remove("hidden");
}

// ── Registration event wiring ────────────────────────────────────────────────

// PID input validation → enable/disable submit
els.inpPid.addEventListener("input", checkRegisterReady);

// Device selection → enable/disable submit
els.selDevice.addEventListener("change", checkRegisterReady);

// Browse folder button
els.btnBrowse.addEventListener("click", async () => {
  els.btnBrowse.disabled = true;
  try {
    const res  = await fetch("/api/browse-dir");
    const data = await res.json();
    if (data.path) els.inpDataDir.value = data.path;
  } catch (e) {
    console.warn("[Browse] fetch failed:", e);
  } finally {
    els.btnBrowse.disabled = false;
  }
});

// Scan button
els.btnScan.addEventListener("click", () => {
  setScanStatus("");
  send({ type: "scan_request" });
});

// Registration form submit
els.btnRegister.addEventListener("click", () => {
  const pid    = els.inpPid.value.trim();
  const session= parseInt(els.inpSession.value, 10) || 1;
  const expId  = els.inpExperimenter.value.trim();
  const dataDir= els.inpDataDir.value.trim() || "data";
  const mac    = isDryRun ? null : (els.selDevice.value || null);

  if (!pid) { els.inpPid.focus(); return; }

  els.btnRegister.disabled = true;
  setConnectStatus("Configuration en cours…");

  send({ type: "register", pid, session, experimenter: expId, data_dir: dataDir, mac });
});

// ── Experimenter overlay ──────────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  if (e.key === "F1")     { e.preventDefault(); els.expOverlay.classList.toggle("hidden"); }
  if (e.key === "Escape") { closeLightbox(); }
});

// ── Overlay button wiring ────────────────────────────────────────────────────

els.btnStart.addEventListener("click",  showConsentScreen);
els.btnConsentAccept.addEventListener("click", submitConsent);
els.btnConsentRefuse.addEventListener("click", refuseConsent);
els.btnPause.addEventListener("click",  () => send({ type: "pause" }));
els.btnResume.addEventListener("click", () => send({ type: "resume" }));
els.btnAbort.addEventListener("click",  () => {
  if (confirm("Interrompre la session en cours ? Les données seront sauvegardées.")) {
    send({ type: "abort" });
  }
});

els.expNotes.addEventListener("blur", flushNotes);

els.btnExpressions.addEventListener("click", openLightbox);
els.lightboxBackdrop.addEventListener("click", closeLightbox);
els.lightboxClose.addEventListener("click", closeLightbox);

els.btnLang.addEventListener("click", () => setLang(lang === "fr" ? "en" : "fr"));

// ── Init ──────────────────────────────────────────────────────────────────────

setWsStatus(false);
setButtons({ start: false, pause: false, resume: false, abort: false });
showScreen("registration");
connect();
