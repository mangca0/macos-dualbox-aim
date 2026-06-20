import random
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional


CMD_CONNECT = 0xAF3C2828
CMD_MOUSE_MOVE = 0xAEDE7345
CMD_MONITOR = 0x27388020

ERR_CREAT_SOCKET = -9000
ERR_NET_RX_TIMEOUT = -8997
ERR_NET_TX = -8998
ERR_NET_CMD = -8996
SUCCESS = 0


@dataclass
class KmboxConfig:
    ip: str = "192.168.2.188"
    port: int = 8888
    mac: str = "1234ABCD"
    socket_timeout: float = 3.0
    connect_attempts: int = 2


class CmdHead:
    def __init__(self):
        self.mac = 0
        self.rand = 0
        self.indexpts = 0
        self.cmd = 0

    def pack(self) -> bytes:
        return struct.pack("<IIII", self.mac, self.rand, self.indexpts, self.cmd)

    @classmethod
    def unpack(cls, data: bytes) -> "CmdHead":
        head = cls()
        head.mac, head.rand, head.indexpts, head.cmd = struct.unpack("<IIII", data[:16])
        return head


class SoftMouse:
    def __init__(self):
        self.button = 0
        self.x = 0
        self.y = 0
        self.wheel = 0
        self.reserved = 0
        self.point = [0] * 7

    def pack(self) -> bytes:
        return struct.pack("<iiiii", self.button, self.x, self.y, self.wheel, self.reserved) + struct.pack("<iiiiiii", *self.point)


class MonitorData:
    def __init__(self):
        self.mouse_left = False
        self.mouse_right = False
        self.mouse_middle = False
        self.mouse_side1 = False
        self.mouse_side2 = False
        self.mouse_x = 0
        self.mouse_y = 0

    @classmethod
    def unpack(cls, data: bytes) -> "MonitorData":
        result = cls()
        if len(data) >= 8:
            buttons = data[1]
            result.mouse_left = bool(buttons & 0x01)
            result.mouse_right = bool(buttons & 0x02)
            result.mouse_middle = bool(buttons & 0x04)
            result.mouse_side1 = bool(buttons & 0x08)
            result.mouse_side2 = bool(buttons & 0x10)
            result.mouse_x, result.mouse_y, _ = struct.unpack("<hhh", data[2:8])
        return result


class KmboxNet:
    def __init__(self, config: Optional[KmboxConfig] = None):
        self.config = config or KmboxConfig()
        self._socket: Optional[socket.socket] = None
        self._initialized = False
        self._indexpts = 0
        self._mac = 0
        self._monitor_socket: Optional[socket.socket] = None
        self._monitor_enabled = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_data = MonitorData()
        self._monitor_lock = threading.Lock()
        self._monitor_callbacks: Dict[str, Callable] = {}

    def init(self) -> int:
        self._initialized = False
        self._mac = self._str_to_hex(self.config.mac)
        attempts = max(1, int(self.config.connect_attempts))
        result = ERR_NET_RX_TIMEOUT
        for _ in range(attempts):
            self._close_command_socket()
            result = self._connect_once()
            if result == SUCCESS:
                self._initialized = True
                return SUCCESS
            if result != ERR_NET_RX_TIMEOUT:
                self._close_command_socket()
                return result
        self._close_command_socket()
        return result

    def _connect_once(self) -> int:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(self.config.socket_timeout)

            head = CmdHead()
            head.mac = self._mac
            head.rand = random.randint(0, 0xFFFFFFFF)
            head.indexpts = 0
            head.cmd = CMD_CONNECT
            self._socket.sendto(head.pack(), (self.config.ip, self.config.port))

            try:
                rx_data, _ = self._socket.recvfrom(1024)
                if len(rx_data) >= 16 and CmdHead.unpack(rx_data).cmd == CMD_CONNECT:
                    return SUCCESS
            except socket.timeout:
                return ERR_NET_RX_TIMEOUT
            return ERR_NET_RX_TIMEOUT
        except Exception:
            return ERR_CREAT_SOCKET

    def close(self):
        self.monitor_stop()
        self._close_command_socket()
        self._initialized = False

    def _close_command_socket(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def mouse_move(self, x: int, y: int) -> int:
        mouse = SoftMouse()
        mouse.x = int(x)
        mouse.y = int(y)
        return self._send_cmd(CMD_MOUSE_MOVE, mouse.pack())

    def monitor_start(self, port: int = 5001) -> int:
        if self._monitor_enabled:
            return SUCCESS
        if port < 1024 or port > 49151:
            return ERR_NET_CMD
        if self._socket is None:
            return ERR_CREAT_SOCKET

        try:
            self._monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._monitor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._monitor_socket.bind(("0.0.0.0", port))
            self._monitor_socket.settimeout(0.1)

            self._indexpts += 1
            head = CmdHead()
            head.mac = self._mac
            head.rand = port | (0xAA55 << 16)
            head.indexpts = self._indexpts
            head.cmd = CMD_MONITOR
            self._socket.sendto(head.pack(), (self.config.ip, self.config.port))

            self._monitor_enabled = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            return SUCCESS
        except Exception:
            return ERR_CREAT_SOCKET

    def monitor_stop(self):
        self._monitor_enabled = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        if self._monitor_socket:
            self._monitor_socket.close()
            self._monitor_socket = None

    def register_callback(self, event: str, callback: Callable):
        self._monitor_callbacks[event] = callback

    def _send_cmd(self, cmd: int, data: bytes = b"") -> int:
        if not self._socket or not self._initialized:
            return ERR_CREAT_SOCKET
        try:
            self._indexpts += 1
            head = CmdHead()
            head.mac = self._mac
            head.rand = random.randint(0, 0xFFFFFFFF)
            head.indexpts = self._indexpts
            head.cmd = cmd
            self._socket.sendto(head.pack() + data, (self.config.ip, self.config.port))
            return SUCCESS
        except Exception:
            return ERR_NET_TX

    def _monitor_loop(self):
        while self._monitor_enabled and self._monitor_socket is not None:
            try:
                data, _ = self._monitor_socket.recvfrom(1024)
                if len(data) < 8:
                    continue
                new_data = MonitorData.unpack(data)
                with self._monitor_lock:
                    old_data = self._monitor_data
                    self._monitor_data = new_data
                self._emit_button_changes(old_data, new_data)
            except socket.timeout:
                continue
            except Exception:
                if self._monitor_enabled:
                    continue

    def _emit_button_changes(self, old: MonitorData, new: MonitorData):
        for button, old_value, new_value in (
            ("left", old.mouse_left, new.mouse_left),
            ("right", old.mouse_right, new.mouse_right),
            ("middle", old.mouse_middle, new.mouse_middle),
            ("side1", old.mouse_side1, new.mouse_side1),
            ("side2", old.mouse_side2, new.mouse_side2),
        ):
            if old_value == new_value:
                continue
            event = f"{button}_{'press' if new_value else 'release'}"
            callback = self._monitor_callbacks.get(event)
            if callback is not None:
                callback(new_value)

    def _str_to_hex(self, value: str) -> int:
        hex_str = "".join(char for char in value if char.isalnum())[:8]
        try:
            return int(hex_str, 16) & 0xFFFFFFFF
        except ValueError:
            return 0x1234ABCD
