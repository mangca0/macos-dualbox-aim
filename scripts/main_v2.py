import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from macos_dualbox_aim.v2 import AimbotConfigV2, AimbotV2, HotkeyConfig, HotkeyMonitor, RealtimeInference, WebTuner


def main():
    config_path = project_root / "configs" / "config_v2.json"
    try:
        if config_path.exists():
            config = AimbotConfigV2.from_json(config_path)
        else:
            config = AimbotConfigV2()
            config.to_json(config_path)
    except ValueError as exc:
        print(f"Invalid config {config_path}: {exc}")
        return

    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    aimbot = AimbotV2(config)
    if not aimbot.connect():
        print("KMBox connection failed after 2 attempts. Check kmbox_ip, kmbox_port, kmbox_mac, power, and network.")
        return

    hotkey = HotkeyMonitor(
        HotkeyConfig(
            trigger_button=config.trigger_button,
            trigger_button_secondary=config.trigger_button_secondary,
            toggle_mode=config.toggle_mode,
            enable_lock_key=config.enable_lock_key,
            lock_key=config.lock_key,
            lock_mode=config.lock_mode,
            kmbox_ip=config.kmbox_ip,
            kmbox_port=config.kmbox_port,
            kmbox_mac=config.kmbox_mac,
            monitor_port=config.monitor_port,
        ),
        aimbot=aimbot,
    )
    if not hotkey.connect():
        print("KMBox hotkey monitor failed. Check monitor_port and KMBox monitor support.")
        aimbot.disconnect()
        return
    hotkey.start()

    engine = RealtimeInference(
        model_path=str(model_path),
        capture_device=config.capture_device,
        target_fps=config.target_fps,
        confidence_threshold=config.detection_confidence_threshold,
        iou_threshold=config.detection_iou_threshold,
        enable_display=config.enable_display,
        crop_size=(config.fov_width, config.fov_height),
        capture_resolution=(config.screen_width, config.screen_height),
        pixel_format=config.pixel_format,
    )

    frame_shape = (config.fov_height, config.fov_width)

    def on_detection(result):
        aimbot.update(result.detections, frame_shape, engine.crop_offset, timing_ms=result.latency_ms)

    engine.on_detection = on_detection
    tuner = None
    if config.enable_tuner:
        try:
            tuner = WebTuner(
                config,
                config_path,
                engine=engine,
                hotkey=hotkey,
                aimbot=aimbot,
                host=config.tuner_host,
                port=config.tuner_port,
            )
            tuner.start()
            print(f"Aimbot V2 tuner running at {tuner.url}")
        except OSError as exc:
            print(f"Aimbot V2 tuner failed to start on {config.tuner_host}:{config.tuner_port}: {exc}")

    print("Aimbot V2 running. Press Ctrl-C to stop.")
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


if __name__ == "__main__":
    main()
