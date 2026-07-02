import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .config import AimbotConfigV64

TUNABLE_FIELDS = {
    "selected_class_ids",
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "slew_limit",
    "max_speed",
    "sensitivity",
    "fov_radius",
    "init_scale",
    "ramp_time",
    "pred_weight_x",
    "pred_weight_y",
    "target_jump_reset",
    "pid_integral_gate_enabled",
    "pid_integral_gate_threshold",
    "pid_integral_gate_rate",
    "stop_brake_enabled",
    "stop_brake_radius",
    "stop_brake_output_decay",
    "stop_brake_pred_decay",
    "stop_brake_min_output",
    "confidence_threshold",
    "iou_threshold",
    "aim_offset_x",
    "aim_offset_y",
    "aim_offset_dynamic",
    "tracker_generate",
    "tracker_terminate",
    "tracker_vx_noise",
    "tracker_vy_noise",
    "tracker_w_noise",
    "tracker_h_noise",
    "tracker_r_std",
    "trigger_button",
    "trigger_button_secondary",
    "crosshair_enabled",
    "crosshair_search_radius",
    "crosshair_min_pixels",
    "crosshair_use_hsv",
    "crosshair_h_min",
    "crosshair_h_max",
    "crosshair_s_min",
    "crosshair_s_max",
    "crosshair_v_min",
    "crosshair_v_max",
    "crosshair_target_r",
    "crosshair_target_g",
    "crosshair_target_b",
    "crosshair_color_tolerance",
}
CONTROLLER_FIELDS = {
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "slew_limit",
    "max_speed",
    "sensitivity",
    "fov_radius",
    "init_scale",
    "ramp_time",
    "pred_weight_x",
    "pred_weight_y",
    "target_jump_reset",
    "pid_integral_gate_enabled",
    "pid_integral_gate_threshold",
    "pid_integral_gate_rate",
    "stop_brake_enabled",
    "stop_brake_radius",
    "stop_brake_output_decay",
    "stop_brake_pred_decay",
    "stop_brake_min_output",
}
TRACKER_FIELDS = {
    "tracker_generate",
    "tracker_terminate",
    "tracker_vx_noise",
    "tracker_vy_noise",
    "tracker_w_noise",
    "tracker_h_noise",
    "tracker_r_std",
}
TRIGGER_BUTTON_OPTIONS = ("left", "right", "side1", "side2")


class WebTuner:
    def __init__(
        self,
        config: AimbotConfigV64,
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
            if TRACKER_FIELDS & set(data):
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
            "config": {field: getattr(self.config, field) for field in sorted(TUNABLE_FIELDS)},
            "dirty": self.dirty,
            "config_path": str(self.config_path),
            "options": {
                "trigger_buttons": list(TRIGGER_BUTTON_OPTIONS),
                "classes": [
                    {"id": index, "name": name}
                    for index, name in enumerate(self.config.class_names or [])
                ],
            },
            "latency": self._latency_snapshot_locked(),
            "aim": self._aim_snapshot_locked(),
            "aim_active": self._aim_active_locked(),
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

    def reset_aim_metrics(self) -> Dict[str, Any]:
        with self.lock:
            if self.aimbot is not None and hasattr(self.aimbot, "reset_aim_metrics"):
                self.aimbot.reset_aim_metrics()
            return self.snapshot_locked()

    def set_aim_active(self, active: bool) -> Dict[str, Any]:
        with self.lock:
            if self.aimbot is None:
                raise ValueError("Aimbot is not attached")
            if self.hotkey is not None and hasattr(self.hotkey, "set_override_active"):
                self.hotkey.set_override_active(bool(active))
            elif bool(active):
                if hasattr(self.aimbot, "activate"):
                    self.aimbot.activate()
            elif hasattr(self.aimbot, "deactivate"):
                self.aimbot.deactivate()
            return self.snapshot_locked()

    def _aim_active_locked(self) -> bool:
        if self.aimbot is None or not hasattr(self.aimbot, "is_active"):
            return False
        return bool(self.aimbot.is_active())

    def _aim_snapshot_locked(self) -> Dict[str, Any]:
        if self.aimbot is None or not hasattr(self.aimbot, "get_aim_metrics_snapshot"):
            return {"available": False, "samples": 0}
        return self.aimbot.get_aim_metrics_snapshot()

    def _validate_trigger_buttons(self, data: Dict[str, Any]):
        if "trigger_button" in data and data["trigger_button"] not in TRIGGER_BUTTON_OPTIONS:
            raise ValueError(f"trigger_button must be one of {list(TRIGGER_BUTTON_OPTIONS)}")
        if "trigger_button_secondary" in data:
            value = data["trigger_button_secondary"]
            if value is not None and value not in TRIGGER_BUTTON_OPTIONS:
                raise ValueError(f"trigger_button_secondary must be one of {list(TRIGGER_BUTTON_OPTIONS)} or null")

    def _apply_runtime_locked(self):
        if self.engine is not None:
            self.engine.confidence_threshold = self.config.confidence_threshold
            self.engine.iou_threshold = self.config.iou_threshold

        if self.hotkey is not None:
            self.hotkey.config.trigger_button = self.config.trigger_button
            self.hotkey.config.trigger_button_secondary = self.config.trigger_button_secondary
            self.hotkey._check_trigger()

        if self.aimbot is not None and hasattr(self.aimbot, "update_selected_classes"):
            self.aimbot.update_selected_classes(self.config.selected_class_ids or [])

        controller = getattr(self.aimbot, "controller", None)
        if controller is not None and hasattr(controller, "update_params"):
            controller.update_params(**self._controller_params_locked())

    def _controller_params_locked(self) -> Dict[str, Any]:
        return {
            "kp": self.config.pid_kp,
            "ki": self.config.pid_ki,
            "kd": self.config.pid_kd,
            "slew_limit": self.config.slew_limit,
            "max_speed": self.config.max_speed,
            "sensitivity": self.config.sensitivity,
            "fov_radius": self.config.fov_radius,
            "init_scale": self.config.init_scale,
            "ramp_time": self.config.ramp_time,
            "pred_weight_x": self.config.pred_weight_x,
            "pred_weight_y": self.config.pred_weight_y,
            "target_jump_reset": self.config.target_jump_reset,
            "pid_integral_gate_enabled": self.config.pid_integral_gate_enabled,
            "pid_integral_gate_threshold": self.config.pid_integral_gate_threshold,
            "pid_integral_gate_rate": self.config.pid_integral_gate_rate,
            "stop_brake_enabled": self.config.stop_brake_enabled,
            "stop_brake_radius": self.config.stop_brake_radius,
            "stop_brake_output_decay": self.config.stop_brake_output_decay,
            "stop_brake_pred_decay": self.config.stop_brake_pred_decay,
            "stop_brake_min_output": self.config.stop_brake_min_output,
        }

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
        if path == "/api/aim":
            self._write_json(HTTPStatus.OK, {"aim": self.server.tuner.snapshot().get("aim", {})})
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
            if path == "/api/aim/reset":
                self._write_json(HTTPStatus.OK, self.server.tuner.reset_aim_metrics())
                return
            if path == "/api/aim/active":
                payload = self._read_json()
                if "active" not in payload:
                    raise ValueError("active is required")
                self._write_json(HTTPStatus.OK, self.server.tuner.set_aim_active(_as_json_bool("active", payload["active"])))
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


def _as_json_bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aimbot V6.4 Tuner</title>
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
    .class-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
    }
    .class-item {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      min-height: 34px;
    }
    .class-item label {
      color: var(--text);
      font-size: 13px;
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
      .class-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Aimbot V6.4 Tuner</h1>
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
      <h2>MPID</h2>
      <div class="field"><label for="pid_kp">Kp</label><input id="pid_kp" data-field="pid_kp" type="range" min="0" max="2" step="0.001"><input data-number-for="pid_kp" type="number" min="0" max="2" step="0.001"></div>
      <div class="field"><label for="pid_ki">Ki</label><input id="pid_ki" data-field="pid_ki" type="range" min="0" max="1" step="0.001"><input data-number-for="pid_ki" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="pid_kd">Kd</label><input id="pid_kd" data-field="pid_kd" type="range" min="0" max="2" step="0.001"><input data-number-for="pid_kd" type="number" min="0" max="2" step="0.001"></div>
      <div class="field"><label for="slew_limit">Slew limit</label><input id="slew_limit" data-field="slew_limit" type="range" min="0" max="200" step="0.1"><input data-number-for="slew_limit" type="number" min="0" max="200" step="0.1"></div>
      <div class="field"><label for="max_speed">Max speed</label><input id="max_speed" data-field="max_speed" type="range" min="1" max="200" step="0.1"><input data-number-for="max_speed" type="number" min="1" max="200" step="0.1"></div>
      <div class="field"><label for="sensitivity">Sensitivity</label><input id="sensitivity" data-field="sensitivity" type="range" min="0.01" max="5" step="0.01"><input data-number-for="sensitivity" type="number" min="0.01" max="5" step="0.01"></div>
      <div class="field check"><label for="pid_integral_gate_enabled">Integral gate</label><input id="pid_integral_gate_enabled" data-field="pid_integral_gate_enabled" type="checkbox"></div>
      <div class="field"><label for="pid_integral_gate_threshold">Integral gate threshold</label><input id="pid_integral_gate_threshold" data-field="pid_integral_gate_threshold" type="range" min="1" max="300" step="0.1"><input data-number-for="pid_integral_gate_threshold" type="number" min="1" max="300" step="0.1"></div>
      <div class="field"><label for="pid_integral_gate_rate">Integral gate rate</label><input id="pid_integral_gate_rate" data-field="pid_integral_gate_rate" type="range" min="0" max="1" step="0.001"><input data-number-for="pid_integral_gate_rate" type="number" min="0" max="1" step="0.001"></div>
    </section>
    <section>
      <h2>Prediction</h2>
      <div class="field"><label for="fov_radius">FOV radius</label><input id="fov_radius" data-field="fov_radius" type="range" min="0" max="1000" step="1"><input data-number-for="fov_radius" type="number" min="0" max="1000" step="1"></div>
      <div class="field"><label for="target_jump_reset">Jump reset</label><input id="target_jump_reset" data-field="target_jump_reset" type="range" min="0" max="300" step="0.1"><input data-number-for="target_jump_reset" type="number" min="0" max="300" step="0.1"></div>
      <div class="field"><label for="init_scale">Init scale</label><input id="init_scale" data-field="init_scale" type="range" min="0.05" max="1" step="0.01"><input data-number-for="init_scale" type="number" min="0.05" max="1" step="0.01"></div>
      <div class="field"><label for="ramp_time">Ramp time</label><input id="ramp_time" data-field="ramp_time" type="range" min="0.001" max="2" step="0.001"><input data-number-for="ramp_time" type="number" min="0.001" max="2" step="0.001"></div>
      <div class="field"><label for="pred_weight_x">Pred weight X</label><input id="pred_weight_x" data-field="pred_weight_x" type="range" min="0" max="1" step="0.01"><input data-number-for="pred_weight_x" type="number" min="0" max="1" step="0.01"></div>
      <div class="field"><label for="pred_weight_y">Pred weight Y</label><input id="pred_weight_y" data-field="pred_weight_y" type="range" min="0" max="1" step="0.01"><input data-number-for="pred_weight_y" type="number" min="0" max="1" step="0.01"></div>
      <div class="field check"><label for="stop_brake_enabled">Stop brake</label><input id="stop_brake_enabled" data-field="stop_brake_enabled" type="checkbox"></div>
      <div class="field"><label for="stop_brake_radius">Stop radius</label><input id="stop_brake_radius" data-field="stop_brake_radius" type="range" min="0" max="80" step="0.1"><input data-number-for="stop_brake_radius" type="number" min="0" max="80" step="0.1"></div>
      <div class="field"><label for="stop_brake_output_decay">Output decay</label><input id="stop_brake_output_decay" data-field="stop_brake_output_decay" type="range" min="0" max="1" step="0.01"><input data-number-for="stop_brake_output_decay" type="number" min="0" max="1" step="0.01"></div>
      <div class="field"><label for="stop_brake_pred_decay">Pred decay</label><input id="stop_brake_pred_decay" data-field="stop_brake_pred_decay" type="range" min="0" max="1" step="0.01"><input data-number-for="stop_brake_pred_decay" type="number" min="0" max="1" step="0.01"></div>
      <div class="field"><label for="stop_brake_min_output">Brake trigger output</label><input id="stop_brake_min_output" data-field="stop_brake_min_output" type="range" min="0" max="200" step="0.1"><input data-number-for="stop_brake_min_output" type="number" min="0" max="200" step="0.1"></div>
    </section>
    <section>
      <h2>Aim</h2>
      <div class="field"><label for="aim_offset_x">Offset X</label><input id="aim_offset_x" data-field="aim_offset_x" type="range" min="-200" max="200" step="0.1"><input data-number-for="aim_offset_x" type="number" min="-200" max="200" step="0.1"></div>
      <div class="field"><label for="aim_offset_y">Offset Y</label><input id="aim_offset_y" data-field="aim_offset_y" type="range" min="-2" max="2" step="0.01"><input data-number-for="aim_offset_y" type="number" min="-2" max="2" step="0.01"></div>
      <div class="field check"><label for="aim_offset_dynamic">Dynamic offset</label><input id="aim_offset_dynamic" data-field="aim_offset_dynamic" type="checkbox"></div>
    </section>
    <section>
      <h2>Crosshair</h2>
      <div class="field check"><label for="crosshair_enabled">Enabled</label><input id="crosshair_enabled" data-field="crosshair_enabled" type="checkbox"></div>
      <div class="field"><label for="crosshair_search_radius">Search radius</label><input id="crosshair_search_radius" data-field="crosshair_search_radius" type="range" min="0" max="300" step="1"><input data-number-for="crosshair_search_radius" type="number" min="0" max="300" step="1"></div>
      <div class="field"><label for="crosshair_min_pixels">Min pixels</label><input id="crosshair_min_pixels" data-field="crosshair_min_pixels" type="range" min="1" max="100" step="1"><input data-number-for="crosshair_min_pixels" type="number" min="1" max="100" step="1"></div>
      <div class="field check"><label for="crosshair_use_hsv">Use HSV</label><input id="crosshair_use_hsv" data-field="crosshair_use_hsv" type="checkbox"></div>
      <div class="field"><label for="crosshair_h_min">H min</label><input id="crosshair_h_min" data-field="crosshair_h_min" type="range" min="0" max="180" step="1"><input data-number-for="crosshair_h_min" type="number" min="0" max="180" step="1"></div>
      <div class="field"><label for="crosshair_h_max">H max</label><input id="crosshair_h_max" data-field="crosshair_h_max" type="range" min="0" max="180" step="1"><input data-number-for="crosshair_h_max" type="number" min="0" max="180" step="1"></div>
      <div class="field"><label for="crosshair_s_min">S min</label><input id="crosshair_s_min" data-field="crosshair_s_min" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_s_min" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_s_max">S max</label><input id="crosshair_s_max" data-field="crosshair_s_max" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_s_max" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_v_min">V min</label><input id="crosshair_v_min" data-field="crosshair_v_min" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_v_min" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_v_max">V max</label><input id="crosshair_v_max" data-field="crosshair_v_max" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_v_max" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_target_r">RGB R</label><input id="crosshair_target_r" data-field="crosshair_target_r" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_target_r" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_target_g">RGB G</label><input id="crosshair_target_g" data-field="crosshair_target_g" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_target_g" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_target_b">RGB B</label><input id="crosshair_target_b" data-field="crosshair_target_b" type="range" min="0" max="255" step="1"><input data-number-for="crosshair_target_b" type="number" min="0" max="255" step="1"></div>
      <div class="field"><label for="crosshair_color_tolerance">RGB tolerance</label><input id="crosshair_color_tolerance" data-field="crosshair_color_tolerance" type="range" min="0" max="255" step="0.1"><input data-number-for="crosshair_color_tolerance" type="number" min="0" max="255" step="0.1"></div>
    </section>
    <section>
      <h2>Detection</h2>
      <div class="field"><label for="confidence_threshold">Confidence</label><input id="confidence_threshold" data-field="confidence_threshold" type="range" min="0" max="1" step="0.001"><input data-number-for="confidence_threshold" type="number" min="0" max="1" step="0.001"></div>
      <div class="field"><label for="iou_threshold">IoU</label><input id="iou_threshold" data-field="iou_threshold" type="range" min="0" max="1" step="0.001"><input data-number-for="iou_threshold" type="number" min="0" max="1" step="0.001"></div>
    </section>
    <section>
      <h2>Classes</h2>
      <div id="class-list" class="class-list"></div>
    </section>
    <section>
      <h2>Tracker</h2>
      <div class="field"><label for="tracker_generate">Generate</label><input id="tracker_generate" data-field="tracker_generate" type="range" min="1" max="8" step="1"><input data-number-for="tracker_generate" type="number" min="1" max="8" step="1"></div>
      <div class="field"><label for="tracker_terminate">Terminate</label><input id="tracker_terminate" data-field="tracker_terminate" type="range" min="1" max="30" step="1"><input data-number-for="tracker_terminate" type="number" min="1" max="30" step="1"></div>
      <div class="field"><label for="tracker_vx_noise">VX noise</label><input id="tracker_vx_noise" data-field="tracker_vx_noise" type="range" min="0.001" max="20" step="0.001"><input data-number-for="tracker_vx_noise" type="number" min="0.001" max="20" step="0.001"></div>
      <div class="field"><label for="tracker_vy_noise">VY noise</label><input id="tracker_vy_noise" data-field="tracker_vy_noise" type="range" min="0.001" max="20" step="0.001"><input data-number-for="tracker_vy_noise" type="number" min="0.001" max="20" step="0.001"></div>
      <div class="field"><label for="tracker_w_noise">W noise</label><input id="tracker_w_noise" data-field="tracker_w_noise" type="range" min="0.001" max="2" step="0.001"><input data-number-for="tracker_w_noise" type="number" min="0.001" max="2" step="0.001"></div>
      <div class="field"><label for="tracker_h_noise">H noise</label><input id="tracker_h_noise" data-field="tracker_h_noise" type="range" min="0.001" max="2" step="0.001"><input data-number-for="tracker_h_noise" type="number" min="0.001" max="2" step="0.001"></div>
      <div class="field"><label for="tracker_r_std">R std</label><input id="tracker_r_std" data-field="tracker_r_std" type="range" min="0.1" max="30" step="0.1"><input data-number-for="tracker_r_std" type="number" min="0.1" max="30" step="0.1"></div>
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
    const integerFields = new Set(["fov_radius", "tracker_generate", "tracker_terminate", "crosshair_search_radius", "crosshair_min_pixels", "crosshair_h_min", "crosshair_h_max", "crosshair_s_min", "crosshair_s_max", "crosshair_v_min", "crosshair_v_max", "crosshair_target_r", "crosshair_target_g", "crosshair_target_b"]);
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
      ["pid_ms", "MPID"],
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
      if (field === "selected_class_ids") return Array.isArray(raw) ? raw.map((value) => Math.trunc(Number(value))) : [];
      if (field === "trigger_button") return String(raw);
      if (field === "trigger_button_secondary") return raw === "" ? null : String(raw);
      if (raw === true || raw === false) return raw;
      const number = Number(raw);
      return integerFields.has(field) ? Math.trunc(number) : number;
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
        if (id === "trigger_button_secondary") {
          const none = document.createElement("option");
          none.value = "";
          none.textContent = "None";
          select.appendChild(none);
        }
        for (const value of state.options.trigger_buttons || []) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = labels[value] || value;
          select.appendChild(option);
        }
      }
    }

    function renderClasses() {
      const target = document.querySelector("#class-list");
      target.innerHTML = "";
      const selected = new Set(state.config.selected_class_ids || []);
      for (const item of state.options.classes || []) {
        const row = document.createElement("div");
        row.className = "class-item";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.id = `class-${item.id}`;
        input.checked = selected.has(item.id);
        input.addEventListener("change", () => {
          const next = Array.from(target.querySelectorAll('input[type="checkbox"]'))
            .filter((element) => element.checked)
            .map((element) => Number(element.dataset.classId));
          scheduleApply("selected_class_ids", next);
        });
        input.dataset.classId = String(item.id);
        const label = document.createElement("label");
        label.htmlFor = input.id;
        label.textContent = `${item.name} (${item.id})`;
        row.appendChild(input);
        row.appendChild(label);
        target.appendChild(row);
      }
    }

    function render(data) {
      state.config = data.config;
      state.options = data.options || {};
      renderLatency(data.latency || {});
      fillHotkeys();
      renderClasses();
      for (const [field, value] of Object.entries(state.config)) {
        const input = document.querySelector(`[data-field="${field}"]`);
        if (!input) continue;
        if (input.type === "checkbox") {
          input.checked = Boolean(value);
        } else {
          input.value = value ?? "";
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
