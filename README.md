# Aimbot V1

干净的 V1 系统雏形，只保留：

- 配置加载
- 实时 CoreML 推理入口
- 检测 bbox 坐标转换
- aim offset
- 四参 PIDF 控制器
- KMBox 热键监控与相对鼠标移动
- Web tuner 实时调参与保存配置

不包含 Kalman、delay compensation、SOT、准星识别、output gain、限幅、deadzone、telemetry。

## 常用命令

```bash
uv sync
uv run python scripts/main_v1.py
uv run python scripts/main_v1_0_1.py
uv run python -m unittest discover -s tests
```

默认配置在 `configs/aimbot_config_v1.json`。
主程序运行后默认在 `http://127.0.0.1:8765` 提供 V1 tuner。

V1.0.1 延迟优化版本入口为 `scripts/main_v1_0_1.py`，配置在
`configs/aimbot_config_v1_0_1.json`。该版本默认 `frame_queue_size=1`，
用于降低 `Queue wait` 的 p95；V1 默认保持 `frame_queue_size=3`。

## 版本目录

当前 V1 实现在 `src/aimbot/v1/`。`scripts/main_v1.py` 和
`scripts/main_v1_0_1.py` 都显式导入 `aimbot.v1`，避免后续 V2 实验影响
V1 主链路。

`src/aimbot/core/` 只放跨 major 版本确认稳定的契约和通用代码。现在先保持
为空，避免把仍在变化的 `inference.py`、`kmbox.py`、控制器或 tuner 提前公共化。

顶层的 `aimbot.config`、`aimbot.controller`、`aimbot.kmbox` 等模块只是兼容层，
旧代码还能继续导入；新代码优先使用 `aimbot.v1.*` 这种显式版本路径。
