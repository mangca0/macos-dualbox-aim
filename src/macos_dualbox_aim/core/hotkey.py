import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from .kmbox import KmboxConfig, KmboxNet, SUCCESS


@dataclass
class HotkeyConfig:
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


@runtime_checkable
class AimbotInterface(Protocol):
    def on_activate(self): ...
    def on_deactivate(self): ...
    def is_active(self) -> bool: ...


class HotkeyMonitor:
    def __init__(self, config: HotkeyConfig, aimbot: Optional[AimbotInterface] = None):
        self.config = config
        self.aimbot = aimbot
        self.kmbox: Optional[KmboxNet] = None
        self.running = False
        self.trigger_active = False
        self.toggle_state = False
        self.lock_active = False
        self.lock_key_pressed = False
        self.thread: Optional[threading.Thread] = None
        self.button_state = {
            "left": False,
            "right": False,
            "middle": False,
            "side1": False,
            "side2": False,
        }
        self.callbacks: list[Callable[[bool], None]] = []

    def connect(self) -> bool:
        self.kmbox = KmboxNet(KmboxConfig(
            ip=self.config.kmbox_ip,
            port=self.config.kmbox_port,
            mac=self.config.kmbox_mac,
        ))
        if self.kmbox.init() != SUCCESS:
            self.kmbox.close()
            self.kmbox = None
            return False
        if self.kmbox.monitor_start(self.config.monitor_port) != SUCCESS:
            self.kmbox.close()
            self.kmbox = None
            return False

        for button in ("left", "right", "middle", "side1", "side2"):
            self.kmbox.register_callback(f"{button}_press", lambda _pressed, name=button: self._on_button_change(name, True))
            self.kmbox.register_callback(f"{button}_release", lambda _pressed, name=button: self._on_button_change(name, False))
        return True

    def disconnect(self):
        self.stop()
        if self.kmbox is not None:
            self.kmbox.close()
            self.kmbox = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def register_state_change_callback(self, callback: Callable[[bool], None]):
        self.callbacks.append(callback)

    def _on_button_change(self, button: str, pressed: bool):
        self.button_state[button] = pressed
        if self.config.enable_lock_key and button == self.config.lock_key:
            self._handle_lock_key(pressed)
        self._check_trigger()

    def _handle_lock_key(self, pressed: bool):
        if self.config.lock_mode == "toggle":
            if pressed and not self.lock_key_pressed:
                self.lock_active = not self.lock_active
            self.lock_key_pressed = pressed
        else:
            self.lock_active = pressed
            self.lock_key_pressed = pressed

    def _check_trigger(self):
        primary_pressed = self.button_state.get(self.config.trigger_button, False)
        secondary_pressed = False
        if self.config.trigger_button_secondary:
            secondary_pressed = self.button_state.get(self.config.trigger_button_secondary, False)

        should_trigger = False if (self.config.enable_lock_key and self.lock_active) else (primary_pressed or secondary_pressed)
        if self.config.toggle_mode:
            if should_trigger and not self.trigger_active:
                self.toggle_state = not self.toggle_state
                self._set_active(self.toggle_state)
        else:
            if should_trigger and not self.trigger_active:
                self._set_active(True)
            elif not should_trigger and self.trigger_active:
                self._set_active(False)
        self.trigger_active = should_trigger

    def _set_active(self, active: bool):
        if self.aimbot is not None:
            if active:
                self.aimbot.on_activate()
            else:
                self.aimbot.on_deactivate()
        for callback in self.callbacks:
            callback(active)

    def _loop(self):
        while self.running:
            time.sleep(0.001)
