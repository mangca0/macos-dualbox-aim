# macos-dualbox-aim V1

干净的 V1 系统雏形，只保留：

- 配置加载
- 实时 CoreML 推理入口
- 检测 bbox 坐标转换
- aim offset
- 四参 PIDF 控制器
- KMBox 热键监控与相对鼠标移动
- Web tuner 实时调参与保存配置

## 常用命令

```bash
uv sync
uv run python scripts/main_v1.py
uv run python -m unittest discover -s tests
```

默认配置在 `configs/config_v1.json`。
主程序运行后默认在 `http://127.0.0.1:8765` 提供 V1 tuner。

## 版本目录

当前 V1 实现在 `src/macos_dualbox_aim/v1/`。`scripts/main_v1.py` 显式导入
V1 版本路径，避免后续 V2 实验影响 V1 主链路。测试也应显式导入对应
major 版本路径。

`src/macos_dualbox_aim/core/` 只放跨 major 版本确认稳定的契约和通用代码。现在先保持
为空，避免把仍在变化的 `inference.py`、`kmbox.py`、控制器或 tuner 提前公共化。

包根目录只保留包入口和版本目录，不再放 `config.py`、`controller.py`、
`kmbox.py` 这类行为模块或兼容壳。新代码统一使用
`macos_dualbox_aim.v1.*`、`macos_dualbox_aim.v2.*` 这种显式版本路径。
