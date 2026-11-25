"use strict";

const $ = (id) => document.getElementById(id);

const statusEl = $("status");
const timerEl = $("timer");
const recordBtn = $("recordBtn");
const recordLabel = $("recordLabel");
const resetBtn = $("resetBtn");
const langSelect = $("langSelect");
const saveAudioChk = $("saveAudioChk");
const errorEl = $("error");
const transcriptEl = $("transcript");
const copyBtn = $("copyBtn");
const downloadBtn = $("downloadBtn");
const metaEl = $("meta");
const canvas = $("waveform");

/* Whisper lokal (transformers.js, WASM) — transkripsi jalan offline di perangkat. */
const WHISPER_MODEL = "Xenova/whisper-tiny";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
const TRANSCRIBE_INTERVAL_MS = 5000;
const MIN_TRANSCRIBE_SEC = 1.2;
const TARGET_RATE = 16000;

const LANG_TO_WHISPER = {
  "id-ID": "id",
  "en-US": "en",
  "en-GB": "en",
  "ja-JP": "ja",
};

let recording = false;
let finalText = "";
let startTime = 0;
let timerHandle = null;

let mediaStream = null;
let audioCtx = null;
let analyser = null;
let drawHandle = null;
let mediaRecorder = null;
const audioChunks = [];
let audioBlob = null;

/* Buffer audio mentah yang belum ditranskripsikan (Float32, sample rate audioCtx). */
let processor = null;
let sourceNode = null;
let pendingChunks = [];
let pendingSamples = 0;
let transcribeTimer = null;
let transcribeBusy = false;

let transcriber = null;
let transcriberLoading = null;

function setStatus(text, state) {
  statusEl.textContent = text;
  if (state) statusEl.dataset.state = state;
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

function fmtTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function renderTranscript() {
  if (!finalText) {
    transcriptEl.innerHTML = '<p class="placeholder">Transkrip akan muncul di sini saat Anda merekam…</p>';
    return;
  }
  transcriptEl.textContent = finalText;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function updateMeta() {
  const words = finalText.trim() ? finalText.trim().split(/\s+/).length : 0;
  metaEl.textContent = `${words} kata${audioBlob ? " · audio tersimpan" : ""}`;
}

/* Waveform */
function startWaveform() {
  const ctx2d = canvas.getContext("2d");
  const draw = () => {
    const w = canvas.width = canvas.clientWidth * devicePixelRatio;
    const h = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx2d.clearRect(0, 0, w, h);
    if (analyser) {
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteTimeDomainData(data);
      ctx2d.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
      ctx2d.lineWidth = 2 * devicePixelRatio;
      ctx2d.beginPath();
      const step = Math.max(1, Math.floor(data.length / 200));
      for (let i = 0, x = 0; i < data.length; i += step, x += w / 200) {
        const y = (data[i] / 255) * h;
        i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
      }
      ctx2d.stroke();
    }
    drawHandle = requestAnimationFrame(draw);
  };
  draw();
}

function stopWaveform() {
  if (drawHandle) cancelAnimationFrame(drawHandle);
  drawHandle = null;
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
}

/* Model Whisper — dimuat sekali, lalu tersimpan di cache browser. */
async function ensureModel() {
  if (transcriber) return transcriber;
  if (transcriberLoading) return transcriberLoading;
  transcriberLoading = (async () => {
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    /* Jangan coba muat model dari jalur lokal aplikasi (di Tauri mengembalikan
       index.html, bukan JSON). Selalu ambil dari Hugging Face lalu cache. */
    env.allowLocalModels = false;
    transcriber = await pipeline("automatic-speech-recognition", WHISPER_MODEL, {
      quantized: true,
      progress_callback: (p) => {
        if (p && p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          setStatus(`Memuat model Whisper… ${pct}% (sekali saja)`, "processing");
        }
      },
    });
    return transcriber;
  })();
  try {
    return await transcriberLoading;
  } catch (err) {
    transcriberLoading = null;
    throw err;
  }
}

/* Audio capture: waveform + buffer 16 kHz untuk Whisper + file audio opsional. */
async function startAudio() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new AudioContext({ sampleRate: TARGET_RATE });
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  sourceNode.connect(analyser);
  startWaveform();

  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);
  processor.onaudioprocess = (e) => {
    const data = e.inputBuffer.getChannelData(0);
    pendingChunks.push(new Float32Array(data));
    pendingSamples += data.length;
  };

  audioChunks.length = 0;
  audioBlob = null;
  if (saveAudioChk.checked && typeof MediaRecorder !== "undefined") {
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => { audioBlob = new Blob(audioChunks, { type: "audio/webm" }); updateMeta(); };
    mediaRecorder.start(1000);
  }
}

function stopAudio() {
  stopWaveform();
  if (processor) {
    processor.onaudioprocess = null;
    processor.disconnect();
    processor = null;
  }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  mediaRecorder = null;
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  mediaStream = null;
  if (audioCtx) audioCtx.close().catch(() => {});
  audioCtx = null;
  analyser = null;
}

function concatChunks(chunks, total) {
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function resampleTo16k(audio, srcRate) {
  if (srcRate === TARGET_RATE || !audio.length) return audio;
  const ratio = srcRate / TARGET_RATE;
  const outLen = Math.floor(audio.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, audio.length - 1);
    out[i] = audio[i0] + (audio[i1] - audio[i0]) * (pos - i0);
  }
  return out;
}

/* Transkripsi potongan audio yang belum diproses (dipanggil berkala & saat berhenti). */
async function transcribePending(force, srcRate) {
  if (!transcriber || transcribeBusy) return;
  if (!pendingSamples) return;
  const rate = srcRate || TARGET_RATE;
  const seconds = pendingSamples / rate;
  if (!force && seconds < MIN_TRANSCRIBE_SEC) return;

  transcribeBusy = true;
  let audio = concatChunks(pendingChunks, pendingSamples);
  pendingChunks = [];
  pendingSamples = 0;
  try {
    audio = resampleTo16k(audio, rate);
    const out = await transcriber(audio, {
      language: LANG_TO_WHISPER[langSelect.value] || "id",
      task: "transcribe",
    });
    const text = (out && out.text ? out.text : "").trim();
    if (text) {
      finalText += (finalText ? " " : "") + text;
      renderTranscript();
      updateMeta();
    }
  } catch (err) {
    showError(`Transkripsi gagal: ${err && err.message ? err.message : err}`);
  } finally {
    transcribeBusy = false;
  }
}

/* Recording control */
async function startRecording() {
  clearError();
  finalText = "";
  pendingChunks = [];
  pendingSamples = 0;
  renderTranscript();
  metaEl.textContent = "";

  setStatus("Menyiapkan model Whisper…", "processing");
  try {
    await ensureModel();
  } catch (err) {
    throw new Error(`Gagal memuat model Whisper (perlu internet sekali untuk mengunduhnya): ${err && err.message ? err.message : err}`);
  }

  await startAudio(); // throws on denied mic -> handled by caller

  recording = true;
  recordBtn.setAttribute("aria-pressed", "true");
  recordLabel.textContent = "Berhenti";
  startTime = Date.now();
  timerHandle = setInterval(() => {
    timerEl.textContent = fmtTime(Math.floor((Date.now() - startTime) / 1000));
  }, 250);

  transcribeTimer = setInterval(() => transcribePending(false, audioCtx ? audioCtx.sampleRate : TARGET_RATE), TRANSCRIBE_INTERVAL_MS);
  setStatus("Merekam — bicara sekarang… (offline, Whisper lokal)", "recording");
}

async function stopRecording() {
  recording = false;
  if (transcribeTimer) { clearInterval(transcribeTimer); transcribeTimer = null; }
  const rate = audioCtx ? audioCtx.sampleRate : TARGET_RATE;

  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop()); // lepas mikrofon lebih awal

  if (pendingSamples > 0) {
    setStatus("Menyelesaikan transkrip…", "processing");
    await transcribePending(true, rate);
  }
  stopAudio();

  clearInterval(timerHandle);
  timerHandle = null;
  renderTranscript();
  updateMeta();
  setStatus("Selesai", "idle");
  recordBtn.setAttribute("aria-pressed", "false");
  recordLabel.textContent = "Mulai Rekam";
}

async function resetAll() {
  if (recording) await stopRecording();
  finalText = "";
  audioBlob = null;
  pendingChunks = [];
  pendingSamples = 0;
  timerEl.textContent = "00:00";
  metaEl.textContent = "";
  renderTranscript();
  setStatus("Siap merekam", "idle");
  clearError();
}

/* Buttons */
recordBtn.addEventListener("click", async () => {
  if (recording) {
    recordBtn.disabled = true;
    try { await stopRecording(); } finally { recordBtn.disabled = false; }
    return;
  }
  recordBtn.disabled = true;
  setStatus("Menunggu izin mikrofon…", "processing");
  try {
    await startRecording();
  } catch (err) {
    if (recording) await stopRecording();
    if (!recording) setStatus("Siap merekam", "idle");
    if (err && err.name === "NotAllowedError") {
      showError("Akses mikrofon ditolak. Izinkan mikrofon di browser lalu coba lagi.");
    } else if (err && err.name === "NotFoundError") {
      showError("Mikrofon tidak ditemukan di perangkat ini.");
    } else {
      showError(`Gagal memulai rekaman: ${err && err.message ? err.message : err}`);
    }
  } finally {
    recordBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => { resetAll(); });

copyBtn.addEventListener("click", async () => {
  if (!finalText.trim()) return;
  try {
    await navigator.clipboard.writeText(finalText.trim());
    copyBtn.textContent = "Tersalin!";
  } catch {
    copyBtn.textContent = "Gagal menyalin";
  }
  setTimeout(() => { copyBtn.textContent = "Salin"; }, 1500);
});

downloadBtn.addEventListener("click", () => {
  if (!finalText.trim()) return;
  const blob = new Blob([finalText.trim()], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `transkrip-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
});

renderTranscript();
