import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from macos_dualbox_aim.core import HotkeyConfig, HotkeyMonitor
from macos_dualbox_aim.v4 import AimbotConfigV4, AimbotV4, WebTuner
from macos_dualbox_aim.v5 import ModelRuntimeConfigV5, RealtimeInferenceV5


def main():
    control_config_path = project_root / "configs" / "config_v4.json"
    model_config_path = project_root / "configs" / "config_v5.json"
    try:
        control_config = load_control_config(control_config_path)
        model_config = load_model_config(model_config_path)
        sync_model_thresholds(control_config, model_config)
    except ValueError as exc:
        print(f"Invalid V5 config: {exc}")
        return

    aimbot = AimbotV4(control_config)
    if not aimbot.connect():
        print("KMBox connection failed after 2 attempts. Check kmbox_ip, kmbox_port, kmbox_mac, power, and network.")
        return

    hotkey = HotkeyMonitor(
        HotkeyConfig(
            trigger_button=control_config.trigger_button,
            trigger_button_secondary=control_config.trigger_button_secondary,
            toggle_mode=control_config.toggle_mode,
            enable_lock_key=control_config.enable_lock_key,
            lock_key=control_config.lock_key,
            lock_mode=control_config.lock_mode,
            kmbox_ip=control_config.kmbox_ip,
            kmbox_port=control_config.kmbox_port,
            kmbox_mac=control_config.kmbox_mac,
            monitor_port=control_config.monitor_port,
        ),
        aimbot=aimbot,
    )
    if not hotkey.connect():
        print("KMBox hotkey monitor failed. Check monitor_port and KMBox monitor support.")
        aimbot.disconnect()
        return
    hotkey.start()

    engine = build_engine(project_root, control_config, model_config)
    frame_shape = (control_config.fov_height, control_config.fov_width)

    def on_detection(result):
        aimbot.update(result.detections, frame_shape, engine.crop_offset, timing_ms=result.latency_ms)

    engine.on_detection = on_detection
    tuner = None
    if control_config.enable_tuner:
        try:
            tuner = WebTuner(
                control_config,
                control_config_path,
                engine=engine,
                hotkey=hotkey,
                aimbot=aimbot,
                host=control_config.tuner_host,
                port=control_config.tuner_port,
            )
            tuner.start()
            print(f"Aimbot V5 tuner running at {tuner.url}")
        except OSError as exc:
            print(f"Aimbot V5 tuner failed to start on {control_config.tuner_host}:{control_config.tuner_port}: {exc}")

    print("Aimbot V5 running with V5 Core ML model runtime. Press Ctrl-C to stop.")
    try:
        engine.start()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        if tuner is not None:
            tuner.stop()
        hotkey.disconnect()
        aimbot.disconnect()


def load_control_config(path: Path) -> AimbotConfigV4:
    if path.exists():
        return AimbotConfigV4.from_json(path)
    config = AimbotConfigV4()
    config.to_json(path)
    return config


def load_model_config(path: Path) -> ModelRuntimeConfigV5:
    if path.exists():
        return ModelRuntimeConfigV5.from_json(path)
    config = ModelRuntimeConfigV5()
    config.to_json(path)
    return config


def sync_model_thresholds(control_config: AimbotConfigV4, model_config: ModelRuntimeConfigV5) -> None:
    control_config.detection_confidence_threshold = model_config.confidence_threshold
    control_config.detection_iou_threshold = model_config.iou_threshold


def build_engine(
    project_root: Path,
    control_config: AimbotConfigV4,
    model_config: ModelRuntimeConfigV5,
) -> RealtimeInferenceV5:
    model_path = Path(model_config.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    return RealtimeInferenceV5(
        model_path=str(model_path),
        class_count=model_config.class_count,
        capture_device=control_config.capture_device,
        target_fps=control_config.target_fps,
        confidence_threshold=model_config.confidence_threshold,
        iou_threshold=model_config.iou_threshold,
        enable_display=control_config.enable_display,
        crop_size=(control_config.fov_width, control_config.fov_height),
        capture_resolution=(control_config.screen_width, control_config.screen_height),
        pixel_format=control_config.pixel_format,
    )


if __name__ == "__main__":
    main()
