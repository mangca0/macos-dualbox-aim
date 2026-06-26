import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

AIMBOT_V62_VERSION = "6.2.0"

BUTTON_NAMES = {"left", "right", "middle", "side1", "side2"}
LOCK_MODES = {"toggle", "hold"}
PIXEL_FORMATS = {"MJPEG", "MJPG", "YUY2", "RGB3", "BGR3", "UYVY"}

INT_FIELDS = {
    "capture_device",
    "class_count",
    "kmbox_port",
    "monitor_port",
    "tuner_port",
    "screen_width",
    "screen_height",
    "fov_width",
    "fov_height",
    "target_fps",
    "fov_radius",
    "tracker_generate",
    "tracker_terminate",
    "crosshair_search_radius",
    "crosshair_min_pixels",
    "crosshair_h_min",
    "crosshair_h_max",
    "crosshair_s_min",
    "crosshair_s_max",
    "crosshair_v_min",
    "crosshair_v_max",
    "crosshair_target_r",
    "crosshair_target_g",
    "crosshair_target_b",
}
FLOAT_FIELDS = {
    "confidence_threshold",
    "iou_threshold",
    "tensor_input_scale",
    "aim_offset_x",
    "aim_offset_y",
    "pid_kp",
    "pid_ki",
    "pid_kd",
    "slew_limit",
    "max_speed",
    "sensitivity",
    "init_scale",
    "ramp_time",
    "pred_weight_x",
    "pred_weight_y",
    "target_jump_reset",
    "pid_integral_gate_threshold",
    "pid_integral_gate_rate",
    "tracker_vx_noise",
    "tracker_vy_noise",
    "tracker_w_noise",
    "tracker_h_noise",
    "tracker_r_std",
    "crosshair_color_tolerance",
}
BOOL_FIELDS = {
    "prefer_image_input",
    "toggle_mode",
    "enable_lock_key",
    "enable_display",
    "debug_mode",
    "aim_offset_dynamic",
    "pid_integral_gate_enabled",
    "enable_tuner",
    "crosshair_enabled",
    "crosshair_use_hsv",
}
STR_FIELDS = {
    "model_path",
    "adapter",
    "trigger_button",
    "lock_key",
    "kmbox_ip",
    "kmbox_mac",
    "pixel_format",
    "lock_mode",
    "tuner_host",
}
OPTIONAL_STR_FIELDS = {"trigger_button_secondary"}


@dataclass
class AimbotConfigV62:
    model_path: str = "models/converted/cs2_fp16_fp16_fast.mlpackage"
    class_count: int = 4
    confidence_threshold: float = 0.65
    iou_threshold: float = 0.3
    adapter: str = "auto"
    prefer_image_input: bool = True
    tensor_input_scale: float = 1.0 / 255.0

    capture_device: int = 0

    trigger_button: str = "right"
    trigger_button_secondary: Optional[str] = "side1"
    toggle_mode: bool = False
    enable_lock_key: bool = False
    lock_key: str = "side2"
    lock_mode: str = "toggle"

    kmbox_ip: str = "192.168.2.188"
    kmbox_port: int = 8888
    kmbox_mac: str = "1234ABCD"
    monitor_port: int = 5001
    enable_tuner: bool = True
    tuner_host: str = "127.0.0.1"
    tuner_port: int = 8765

    screen_width: int = 1920
    screen_height: int = 1080
    pixel_format: str = "MJPEG"
    fov_width: int = 320
    fov_height: int = 320

    target_fps: int = 240
    enable_display: bool = False
    debug_mode: bool = False

    aim_offset_x: float = 0.0
    aim_offset_y: float = -0.35
    aim_offset_dynamic: bool = True

    pid_kp: float = 0.37
    pid_ki: float = 0.0
    pid_kd: float = 0.0
    slew_limit: float = 40.0
    max_speed: float = 30.0
    sensitivity: float = 1.0
    fov_radius: int = 256
    init_scale: float = 0.6
    ramp_time: float = 0.5
    pred_weight_x: float = 0.5
    pred_weight_y: float = 0.5
    target_jump_reset: float = 40.0
    pid_integral_gate_enabled: bool = True
    pid_integral_gate_threshold: float = 50.0
    pid_integral_gate_rate: float = 0.025

    tracker_generate: int = 2
    tracker_terminate: int = 8
    tracker_vx_noise: float = 1.0
    tracker_vy_noise: float = 1.0
    tracker_w_noise: float = 0.01
    tracker_h_noise: float = 0.01
    tracker_r_std: float = 5.0

    crosshair_enabled: bool = True
    crosshair_search_radius: int = 80
    crosshair_min_pixels: int = 3
    crosshair_use_hsv: bool = True
    crosshair_h_min: int = 50
    crosshair_h_max: int = 80
    crosshair_s_min: int = 120
    crosshair_s_max: int = 255
    crosshair_v_min: int = 120
    crosshair_v_max: int = 255
    crosshair_target_r: int = 0
    crosshair_target_g: int = 255
    crosshair_target_b: int = 0
    crosshair_color_tolerance: float = 30.0

    @property
    def version(self) -> str:
        return AIMBOT_V62_VERSION

    @classmethod
    def from_json(cls, path: str | Path) -> "AimbotConfigV62":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError("Config root must be a JSON object")

        config = cls()
        field_names = {item.name for item in fields(cls)}
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if key not in field_names:
                raise ValueError(f"Unknown config field: {key}")
            setattr(config, key, _coerce_config_value(key, value))
        config.validate()
        return config

    def validate(self):
        if self.trigger_button not in BUTTON_NAMES:
            raise ValueError(f"trigger_button must be one of {sorted(BUTTON_NAMES)}")
        if self.trigger_button_secondary is not None and self.trigger_button_secondary not in BUTTON_NAMES:
            raise ValueError(f"trigger_button_secondary must be one of {sorted(BUTTON_NAMES)} or null")
        if self.lock_key not in BUTTON_NAMES:
            raise ValueError(f"lock_key must be one of {sorted(BUTTON_NAMES)}")
        if self.lock_mode not in LOCK_MODES:
            raise ValueError(f"lock_mode must be one of {sorted(LOCK_MODES)}")
        if self.pixel_format not in PIXEL_FORMATS:
            raise ValueError(f"pixel_format must be one of {sorted(PIXEL_FORMATS)}")
        if self.adapter not in {"auto"}:
            raise ValueError("adapter must be auto")
        if self.class_count <= 0:
            raise ValueError("class_count must be positive")
        if not 1 <= self.kmbox_port <= 65535:
            raise ValueError("kmbox_port must be between 1 and 65535")
        if not 1024 <= self.monitor_port <= 49151:
            raise ValueError("monitor_port must be between 1024 and 49151")
        if not 1024 <= self.tuner_port <= 65535:
            raise ValueError("tuner_port must be between 1024 and 65535")
        for name in ("screen_width", "screen_height", "fov_width", "fov_height", "target_fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fov_width > self.screen_width or self.fov_height > self.screen_height:
            raise ValueError("fov_width/fov_height must not exceed screen_width/screen_height")
        for name in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.tensor_input_scale <= 0.0:
            raise ValueError("tensor_input_scale must be positive")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be positive")
        if self.fov_radius < 0:
            raise ValueError("fov_radius must be zero or positive")
        if not 0.05 <= self.init_scale <= 1.0:
            raise ValueError("init_scale must be between 0.05 and 1.0")
        if self.ramp_time <= 0.0:
            raise ValueError("ramp_time must be positive")
        for name in ("pred_weight_x", "pred_weight_y"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.target_jump_reset < 0.0:
            raise ValueError("target_jump_reset must be zero or positive")
        if self.pid_integral_gate_threshold <= 0.0:
            raise ValueError("pid_integral_gate_threshold must be positive")
        if not 0.0 <= self.pid_integral_gate_rate <= 1.0:
            raise ValueError("pid_integral_gate_rate must be between 0.0 and 1.0")
        for name in ("tracker_generate", "tracker_terminate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("tracker_vx_noise", "tracker_vy_noise", "tracker_w_noise", "tracker_h_noise", "tracker_r_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.crosshair_search_radius < 0:
            raise ValueError("crosshair_search_radius must be zero or positive")
        if self.crosshair_min_pixels <= 0:
            raise ValueError("crosshair_min_pixels must be positive")
        for name in ("crosshair_h_min", "crosshair_h_max"):
            value = getattr(self, name)
            if not 0 <= value <= 180:
                raise ValueError(f"{name} must be between 0 and 180")
        for name in (
            "crosshair_s_min",
            "crosshair_s_max",
            "crosshair_v_min",
            "crosshair_v_max",
            "crosshair_target_r",
            "crosshair_target_g",
            "crosshair_target_b",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        if self.crosshair_s_min > self.crosshair_s_max:
            raise ValueError("crosshair_s_min must not exceed crosshair_s_max")
        if self.crosshair_v_min > self.crosshair_v_max:
            raise ValueError("crosshair_v_min must not exceed crosshair_v_max")
        if self.crosshair_color_tolerance < 0.0:
            raise ValueError("crosshair_color_tolerance must be zero or positive")

    def update_from_mapping(self, data: Mapping[str, Any], allowed_fields: Optional[Set[str]] = None):
        field_names = {item.name for item in fields(type(self))}
        current = asdict(self)

        for key, value in data.items():
            if key.startswith("_"):
                continue
            if key not in field_names:
                raise ValueError(f"Unknown config field: {key}")
            if allowed_fields is not None and key not in allowed_fields:
                raise ValueError(f"Field is not runtime-tunable: {key}")
            current[key] = _coerce_config_value(key, value)

        candidate = type(self)(**current)
        candidate.validate()
        for key in current:
            setattr(self, key, getattr(candidate, key))

    def to_json(self, path: str | Path):
        data: Dict[str, Any] = {
            "_comment": "macos-dualbox-aim V6.2.0 - V6.1 plus color-detected crosshair aim reference",
            "_version": AIMBOT_V62_VERSION,
            "_description": "capture card -> CoreML detections -> crosshair color detection -> target relative to crosshair -> learned MPID -> KMBox",
        }
        output = Path(path)
        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    data.update({
                        key: value
                        for key, value in existing.items()
                        if key.startswith("_") and key not in {"_comment", "_version", "_description"}
                    })
            except (OSError, json.JSONDecodeError):
                pass
        data["_version"] = AIMBOT_V62_VERSION
        data.update(asdict(self))

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def _coerce_config_value(key: str, value: Any) -> Any:
    if key in INT_FIELDS:
        return _as_int(key, value)
    if key in FLOAT_FIELDS:
        return _as_float(key, value)
    if key in BOOL_FIELDS:
        return _as_bool(key, value)
    if key in STR_FIELDS:
        return _as_str(key, value)
    if key in OPTIONAL_STR_FIELDS:
        if value is None:
            return None
        return _as_str(key, value)
    raise ValueError(f"Unsupported config field: {key}")


def _as_int(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _as_float(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _as_bool(key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _as_str(key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
