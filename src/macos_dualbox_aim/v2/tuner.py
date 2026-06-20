import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .config import AimbotConfigV2

TUNABLE_FIELDS = {
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "pid_kf",
    "enable_kalman_filter",
    "kalman_process_noise",
    "kalman_measurement_noise",
    "kalman_initial_covariance",
    "detection_confidence_threshold",
    "detection_iou_threshold",
    "target_classes",
    "class_priority_weights",
    "aim_offset_x",
    "aim_offset_y",
    "aim_offset_dynamic",
    "trigger_button",
    "trigger_button_secondary",
}
KALMAN_FIELDS = {
    "enable_kalman_filter",
    "kalman_process_noise",
    "kalman_measurement_noise",
    "kalman_initial_covariance",
}
TRIGGER_BUTTON_OPTIONS = ("left", "right", "side1", "side2")


class WebTuner:
    def __init__(
        self,
        config: AimbotConfigV2,
        config_path: str | Path,
        *,
        engine: Optional[Any] = None,
        hotkey: Optional[Any] = None,
        aimbot: Optional[Any] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.config = config
        self.config_path = Path(config_path)
        self.engine = engine
        self.hotkey = hotkey
        self.aimbot = aimbot
        self.host = config.tuner_host if host is None else host
        self.port = config.tuner_port if port is None else int(port)
        self.lock = threading.RLock()
        self.server: Optional[_TunerHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.dirty = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        if self.server is not None:
            return
        self.server = _TunerHTTPServer((self.host, self.port), _TunerRequestHandler)
        self.server.tuner = self
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
        self.server = None

    def update_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        with self.lock:
            self._validate_trigger_buttons(data)
            self.config.update_from_mapping(data, allowed_fields=TUNABLE_FIELDS)
            self._apply_runtime_locked()
            if KALMAN_FIELDS & set(data):
                self._reset_aimbot_tracking_locked()
            self.dirty = True
            return self.snapshot_locked()

    def save_config(self) -> Dict[str, Any]:
        with self.lock:
            self.config.to_json(self.config_path)
            self.dirty = False
            return self.snapshot_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> Dict[str, Any]:
        return {
            "config": {
                "pid_kp": self.config.pid_kp,
                "pid_ki": self.config.pid_ki,
                "pid_kd": self.config.pid_kd,
                "pid_kf": self.config.pid_kf,
                "enable_kalman_filter": self.config.enable_kalman_filter,
                "kalman_process_noise": self.config.kalman_process_noise,
                "kalman_measurement_noise": self.config.kalman_measurement_noise,
                "kalman_initial_covariance": self.config.kalman_initial_covariance,
                "detection_confidence_threshold": self.config.detection_confidence_threshold,
                "detection_iou_threshold": self.config.detection_iou_threshold,
                "target_classes": self.config.target_classes,
                "class_priority_weights": {
                    str(key): value for key, value in self.config.class_priority_weights.items()
                },
                "aim_offset_x": self.config.aim_offset_x,
                "aim_offset_y": self.config.aim_offset_y,
                "aim_offset_dynamic": self.config.aim_offset_dynamic,
                "trigger_button": self.config.trigger_button,
                "trigger_button_secondary": self.config.trigger_button_secondary,
            },
            "dirty": self.dirty,
            "config_path": str(self.config_path),
            "options": {
                "trigger_buttons": list(TRIGGER_BUTTON_OPTIONS),
            },
            "latency": self._latency_snapshot_locked(),
        }

    def _latency_snapshot_locked(self) -> Dict[str, Any]:
        if self.engine is None or not hasattr(self.engine, "get_latency_snapshot"):
            return {
                "available": False,
                "fps": 0.0,
                "window": 0,
                "current": {},
                "avg": {},
                "p95": {},
                "max": {},
                "counters": {},
            }
        return self.engine.get_latency_snapshot()

    def _validate_trigger_buttons(self, data: Dict[str, Any]):
        for key in ("trigger_button", "trigger_button_secondary"):
            if key not in data:
                continue
            value = data[key]
            if value not in TRIGGER_BUTTON_OPTIONS:
                raise ValueError(f"{key} must be one of {list(TRIGGER_BUTTON_OPTIONS)}")

    def _apply_runtime_locked(self):
        if self.engine is not None:
            self.engine.confidence_threshold = self.config.detection_confidence_threshold
            self.engine.iou_threshold = self.config.detection_iou_threshold

        if self.hotkey is not None:
            self.hotkey.config.trigger_button = self.config.trigger_button
            self.hotkey.config.trigger_button_secondary = self.config.trigger_button_secondary
            self.hotkey._check_trigger()

    def _reset_aimbot_tracking_locked(self):
        if self.aimbot is not None and hasattr(self.aimbot, "reset_tracking"):
            self.aimbot.reset_tracking()


class _TunerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    tuner: WebTuner


class _TunerRequestHandler(BaseHTTPRequestHandler):
    server: _TunerHTTPServer

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._write_html(_HTML)
            return
        if path == "/api/config":
            self._write_json(HTTPStatus.OK, self.server.tuner.snapshot())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/config":
                payload = self._read_json()
                self._write_json(HTTPStatus.OK, self.server.tuner.update_config(payload))
                return
            if path == "/api/save":
                self._write_json(HTTPStatus.OK, self.server.tuner.save_config())
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, _format: str, *_args):
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body") from exc
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def _write_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, status: HTTPStatus, data: Dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aimbot V2 Tuner</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --line: #d7d9d2;
      --text: #222623;
      --muted: #667069;
      --accent: #067a75;
      --accent-dark: #055f5b;
      --save: #9a5b00;
      --error: #b3261e;
      --ok: #276738;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.35;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 1;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 720;
      letter-spacing: 0;
    }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .field {
      display: grid;
      grid-template-columns: minmax(120px, 0.8fr) minmax(140px, 1fr) 88px;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      margin: 9px 0;
    }
    label {
      font-size: 13px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    input, select, textarea, button {
      font: inherit;
      letter-spacing: 0;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    input[type="number"], select, textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      background: #fff;
      padding: 7px 8px;
    }
    textarea {
      resize: vertical;
      min-height: 68px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      grid-column: 2 / 4;
    }
    .check {
      grid-template-columns: minmax(120px, 0.8fr) minmax(140px, 1fr);
    }
    .check input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }
    .value {
      min-width: 0;
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 6px;
      color: #fff;
      background: var(--accent);
      padding: 8px 12px;
      min-height: 36px;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    #save { background: var(--save); }
    #status {
      font-size: 13px;
      color: var(--muted);
      min-width: min(260px, 100%);
      overflow-wrap: anywhere;
    }
    #status.error { color: var(--error); }
    #status.ok { color: var(--ok); }
    .full { grid-column: 1 / -1; }
    .metric-head, .metric-row {
      display: grid;
      grid-template-columns: minmax(110px, 1fr) repeat(4, minmax(72px, 0.7fr));
      gap: 8px;
      align-items: center;
      min-height: 30px;
      font-variant-numeric: tabular-nums;
    }
    .metric-head {
      color: var(--muted);
      font-size: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 6px;
      margin-bottom: 4px;
    }
    .metric-row {
      font-size: 13px;
      border-bottom: 1px solid #eef0ea;
    }
    .metric-row strong {
      color: var(--text);
      font-weight: 680;
    }
    .metric-row span:not(:first-child), .metric-head span:not(:first-child) {
      text-align: right;
    }
    .metric-summary {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
      font-variant-numeric: tabular-nums;
    }
    .highlight-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px 16px;
      padding: 8px 0 16px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 12px;
    }
    .highlight {
      min-width: 0;
    }
    .highlight-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .highlight-main {
      font-size: 20px;
      font-weight: 720;
      font-variant-numeric: tabular-nums;
    }
    .highlight-sub {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      margin-top: 2px;
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; padding: 12px; }
      .field { grid-template-columns: 1fr; gap: 6px; }
      textarea { grid-column: auto; }
      .value { text-align: left; }
      .metric-head, .metric-row {
        grid-template-columns: minmax(90px, 1fr) repeat(4, minmax(56px, 0.7fr));
        gap: 5px;
        font-size: 12px;
      }
      .highlight-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Aimbot V2 Tuner</h1>
    <div class="actions">
      <button id="reload" type="button">Reload</button>
      <button id="save" type="button">Save config</button>
      <span id="status">Loading...</span>
    </div>
  </header>
  <main>
    <section class="full">
      <h2>Latency</h2>
      <div class="metric-summary">
        <span>FPS <strong id="latency-fps">--</strong></span>
        <span>Window <strong id="latency-window">0</strong></span>
        <span>Captured <strong id="frames-captured">0</strong></span>
        <span>Inferred <strong id="frames-inferred">0</strong></span>
        <span>Dropped <strong id="frames-dropped">0</strong></span>
        <span>Replaced <strong id="frames-replaced">0</strong></span>
        <span>Drained <strong id="frames-drained">0</strong></span>
      </div>
      <div id="latency-highlights" class="highlight-grid"></div>
      <div class="metric-head"><span>Stage</span><span>Now</span><span>Avg</span><span>P95</span><span>Max</span></div>
      <div id="latency-rows"></div>
    </section>
    <section>
      <h2>PIDF</h2>
      <div class="field"><label for="pid_kp">Kp</label><input id="pid_kp" data-field="pid_kp" type="range" min="0" max="3" step="0.001"><input data-number-for="pid_kp" type="number" min="0" max="3" step="0.001"></div>
      <div class="field"><label for="pid_ki">Ki</label><input id="pid_ki" data-field="pid_ki" type="range" min="0" max="1" step="0.001"><input data-number-for="pid_ki" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="pid_kd">Kd</label><input id="pid_kd" data-field="pid_kd" type="range" min="0" max="1" step="0.001"><input data-number-for="pid_kd" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="pid_kf">Kf</label><input id="pid_kf" data-field="pid_kf" type="range" min="-1" max="1" step="0.001"><input data-number-for="pid_kf" type="number" min="-1" max="1" step="0.001"></div>
    </section>
    <section>
      <h2>Kalman</h2>
      <div class="field check"><label for="enable_kalman_filter">Enabled</label><input id="enable_kalman_filter" data-field="enable_kalman_filter" type="checkbox"></div>
      <div class="field"><label for="kalman_process_noise">Process</label><input id="kalman_process_noise" data-field="kalman_process_noise" type="range" min="0.001" max="500" step="0.001"><input data-number-for="kalman_process_noise" type="number" min="0.001" max="500" step="0.001"></div>
      <div class="field"><label for="kalman_measurement_noise">Measure</label><input id="kalman_measurement_noise" data-field="kalman_measurement_noise" type="range" min="0.001" max="1000" step="0.001"><input data-number-for="kalman_measurement_noise" type="number" min="0.001" max="1000" step="0.001"></div>
      <div class="field"><label for="kalman_initial_covariance">Initial cov</label><input id="kalman_initial_covariance" data-field="kalman_initial_covariance" type="range" min="0.001" max="5000" step="0.001"><input data-number-for="kalman_initial_covariance" type="number" min="0.001" max="5000" step="0.001"></div>
    </section>
    <section>
      <h2>Detection</h2>
      <div class="field"><label for="detection_confidence_threshold">Confidence</label><input id="detection_confidence_threshold" data-field="detection_confidence_threshold" type="range" min="0" max="1" step="0.001"><input data-number-for="detection_confidence_threshold" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="detection_iou_threshold">IoU</label><input id="detection_iou_threshold" data-field="detection_iou_threshold" type="range" min="0" max="1" step="0.001"><input data-number-for="detection_iou_threshold" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="target_classes">Target classes</label><input id="target_classes" data-field="target_classes" type="text"><span class="value"></span></div>
      <div class="field"><label for="class_priority_weights">Class weights</label><textarea id="class_priority_weights" data-field="class_priority_weights"></textarea></div>
    </section>
    <section>
      <h2>Aim Offset</h2>
      <div class="field"><label for="aim_offset_x">Offset X</label><input id="aim_offset_x" data-field="aim_offset_x" type="range" min="-200" max="200" step="0.1"><input data-number-for="aim_offset_x" type="number" min="-200" max="200" step="0.1"></div>
      <div class="field"><label for="aim_offset_y">Offset Y</label><input id="aim_offset_y" data-field="aim_offset_y" type="range" min="-2" max="2" step="0.001"><input data-number-for="aim_offset_y" type="number" min="-2" max="2" step="0.001"></div>
      <div class="field check"><label for="aim_offset_dynamic">Dynamic Y</label><input id="aim_offset_dynamic" data-field="aim_offset_dynamic" type="checkbox"></div>
    </section>
    <section>
      <h2>Hotkeys</h2>
      <div class="field"><label for="trigger_button">Trigger</label><select id="trigger_button" data-field="trigger_button"></select><span class="value"></span></div>
      <div class="field"><label for="trigger_button_secondary">Trigger secondary</label><select id="trigger_button_secondary" data-field="trigger_button_secondary"></select><span class="value"></span></div>
    </section>
  </main>
  <script>
    const statusEl = document.querySelector("#status");
    const state = { config: {}, options: {}, timers: {} };
    const labels = { left: "Mouse left", right: "Mouse right", side1: "Upper side", side2: "Lower side" };
    const metricOrder = [
      ["read_included_total_ms", "Read-included total"],
      ["program_total_ms", "Program total"],
      ["capture_read_ms", "Capture read"],
      ["crop_ms", "Crop"],
      ["queue_wait_ms", "Queue wait"],
      ["preprocess_ms", "Preprocess"],
      ["coreml_ms", "CoreML"],
      ["postprocess_ms", "Postprocess"],
      ["inference_ms", "Inference"],
      ["detection_callback_ms", "Detection callback"],
      ["target_select_ms", "Target select"],
      ["pid_ms", "PID"],
      ["kmbox_send_ack_ms", "KMBox send/ack"]
    ];
    const highlightOrder = [
      ["program_total_ms", "Program total"],
      ["queue_wait_ms", "Queue wait"],
      ["inference_ms", "Inference"],
      ["coreml_ms", "CoreML"],
      ["dropped", "Dropped"],
      ["kmbox_send_ack_ms", "KMBox send/ack"]
    ];

    function setStatus(text, kind = "") {
      statusEl.textContent = text;
      statusEl.className = kind;
    }

    async function request(path, body) {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function normalizeValue(field, raw) {
      if (field === "aim_offset_dynamic" || field === "enable_kalman_filter") return Boolean(raw);
      if (field === "target_classes") {
        if (Array.isArray(raw)) return raw;
        return String(raw).split(",").map((item) => item.trim()).filter(Boolean).map((item) => Number.parseInt(item, 10));
      }
      if (field === "class_priority_weights") {
        return typeof raw === "string" ? JSON.parse(raw) : raw;
      }
      if (field === "trigger_button" || field === "trigger_button_secondary") return String(raw);
      return Number(raw);
    }

    async function applyField(field, value) {
      const payload = {};
      payload[field] = normalizeValue(field, value);
      const data = await request("/api/config", payload);
      state.config = data.config;
      setStatus(data.dirty ? "Live, unsaved" : "Live", "ok");
    }

    function scheduleApply(field, value) {
      window.clearTimeout(state.timers[field]);
      state.timers[field] = window.setTimeout(async () => {
        try {
          await applyField(field, value);
        } catch (error) {
          setStatus(error.message, "error");
        }
      }, 120);
    }

    function fillHotkeys() {
      for (const id of ["trigger_button", "trigger_button_secondary"]) {
        const select = document.querySelector(`#${id}`);
        select.innerHTML = "";
        for (const value of state.options.trigger_buttons || []) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = labels[value] || value;
          select.appendChild(option);
        }
      }
    }

    function render(data) {
      state.config = data.config;
      state.options = data.options || {};
      renderLatency(data.latency || {});
      fillHotkeys();
      for (const [field, value] of Object.entries(state.config)) {
        const input = document.querySelector(`[data-field="${field}"]`);
        if (!input) continue;
        if (input.type === "checkbox") {
          input.checked = Boolean(value);
        } else if (field === "target_classes") {
          input.value = Array.isArray(value) ? value.join(",") : "";
        } else if (field === "class_priority_weights") {
          input.value = JSON.stringify(value, null, 2);
        } else {
          input.value = value;
        }
        const number = document.querySelector(`[data-number-for="${field}"]`);
        if (number) number.value = value;
      }
      setStatus(data.dirty ? "Live, unsaved" : "Loaded", data.dirty ? "ok" : "");
    }

    function formatMs(value) {
      if (!Number.isFinite(value)) return "--";
      if (value >= 100) return value.toFixed(1);
      return value.toFixed(2);
    }

    function formatCount(value) {
      return Number.isFinite(value) ? String(value) : "0";
    }

    function renderLatency(latency) {
      document.querySelector("#latency-fps").textContent = Number.isFinite(latency.fps) ? latency.fps.toFixed(1) : "--";
      document.querySelector("#latency-window").textContent = String(latency.window || 0);
      const counters = latency.counters || {};
      document.querySelector("#frames-captured").textContent = String(counters.frames_captured || 0);
      document.querySelector("#frames-inferred").textContent = String(counters.frames_inferred || 0);
      document.querySelector("#frames-dropped").textContent = String(counters.frames_dropped || 0);
      document.querySelector("#frames-replaced").textContent = String(counters.frame_queue_replaced || 0);
      document.querySelector("#frames-drained").textContent = String(counters.frame_queue_drained || 0);
      renderHighlights(latency, counters);
      const rows = document.querySelector("#latency-rows");
      rows.innerHTML = "";
      for (const [key, label] of metricOrder) {
        const row = document.createElement("div");
        row.className = "metric-row";
        const current = latency.current || {};
        const avg = latency.avg || {};
        const p95 = latency.p95 || {};
        const max = latency.max || {};
        row.innerHTML = `<span><strong>${label}</strong></span><span>${formatMs(current[key])}</span><span>${formatMs(avg[key])}</span><span>${formatMs(p95[key])}</span><span>${formatMs(max[key])}</span>`;
        rows.appendChild(row);
      }
    }

    function renderHighlights(latency, counters) {
      const target = document.querySelector("#latency-highlights");
      const current = latency.current || {};
      const avg = latency.avg || {};
      const p95 = latency.p95 || {};
      target.innerHTML = "";
      for (const [key, label] of highlightOrder) {
        const item = document.createElement("div");
        item.className = "highlight";
        if (key === "dropped") {
          const captured = Number(counters.frames_captured || 0);
          const dropped = Number(counters.frames_dropped || 0);
          const rate = captured > 0 ? dropped * 100 / captured : 0;
          item.innerHTML = `<div class="highlight-label">${label}</div><div class="highlight-main">${formatCount(dropped)}</div><div class="highlight-sub">${rate.toFixed(1)}% of captured</div>`;
        } else {
          item.innerHTML = `<div class="highlight-label">${label}</div><div class="highlight-main">${formatMs(avg[key])} ms</div><div class="highlight-sub">now ${formatMs(current[key])} / p95 ${formatMs(p95[key])}</div>`;
        }
        target.appendChild(item);
      }
    }

    function bindInputs() {
      document.querySelectorAll("[data-field]").forEach((input) => {
        const field = input.dataset.field;
        const eventName = input.tagName === "TEXTAREA" || input.type === "text" || input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input";
        input.addEventListener(eventName, () => {
          const value = input.type === "checkbox" ? input.checked : input.value;
          const number = document.querySelector(`[data-number-for="${field}"]`);
          if (number) number.value = value;
          scheduleApply(field, value);
        });
      });
      document.querySelectorAll("[data-number-for]").forEach((input) => {
        const field = input.dataset.numberFor;
        input.addEventListener("input", () => {
          const range = document.querySelector(`[data-field="${field}"]`);
          if (range) range.value = input.value;
          scheduleApply(field, input.value);
        });
      });
      document.querySelector("#reload").addEventListener("click", load);
      document.querySelector("#save").addEventListener("click", async () => {
        try {
          const data = await request("/api/save", {});
          render(data);
          setStatus("Saved", "ok");
        } catch (error) {
          setStatus(error.message, "error");
        }
      });
    }

    async function load() {
      try {
        render(await request("/api/config"));
      } catch (error) {
        setStatus(error.message, "error");
      }
    }

    async function refreshLatency() {
      try {
        const data = await request("/api/config");
        renderLatency(data.latency || {});
      } catch (_error) {
      }
    }

    bindInputs();
    load();
    window.setInterval(refreshLatency, 500);
  </script>
</body>
</html>
"""
