# macos-dualbox-aim V1.2.0

V1.2.0 是基于 V1.0.0 行为链路的延迟观测和采集模式实验版本，只保留：

- 配置加载
- 实时 CoreML 推理入口
- 检测 bbox 坐标转换
- aim offset
- 四参 PIDF 控制器
- KMBox 热键监控与相对鼠标移动
- Web tuner 实时调参与保存配置

## V1.2.0 延迟分析

V1.2.0 保留 V1.0.0 的运行链路行为，并在 V1.1.1 的采集链路细分诊断上，新增独立采集模式 probe 工具，用于比较不同色彩格式、FPS、分辨率和后端组合：

- `capture_grab_ms`
- `capture_retrieve_ms`
- `capture_frame_interval_ms`
- 采集后端、实际分辨率/FPS/FourCC/缓冲设置
- `capture_grab_failures`、`capture_retrieve_failures`
- `scripts/capture_probe.py` 矩阵测试并输出 Markdown/JSONL

V1.1.0 已验证的一轮微优化没有带来实际可感知的平均延迟改善，记录见
`docs/latency-optimization-attempts.md`。

## 常用命令

```bash
uv sync
uv run python scripts/main_v1.py
uv run python -m unittest discover -s tests
```

默认配置在 `configs/config_v1.json`。
主程序运行后默认在 `http://127.0.0.1:8765` 提供 V1 tuner。

## 延迟分析工具

先启动主程序，确认 tuner 在运行。每次采集会从
`http://127.0.0.1:8765/api/config` 拉取 tuner 延迟快照，并写入 `latency_runs/`。

```bash
uv run python scripts/main_v1.py

for i in 1 2 3 4 5; do
  uv run python scripts/latency_tool.py capture --label v1.2.0 --run run$i --duration 60 --interval 0.5
done

uv run python scripts/latency_tool.py compare "latency_runs/*.jsonl" --baseline-label v1.0.0 --candidate-label v1.2.0 --out latency_runs/v1.0.0_vs_v1.2.0.md
```

对比报告里的主平均值使用 tuner 返回的 `latency.avg`，也就是每次抓到的 tuner 滚动平均值再做 run 间汇总；负数 delta/change 表示 candidate 更快。
采集工具会检查 `--label` 是否匹配 tuner 返回的 runtime 版本，避免误标。跨版本对比时，先切到对应 tag/分支运行主程序并用匹配的 label 采集，再回到当前版本生成 compare 报告。

## 采集模式 probe

先停止主程序，避免两个进程同时占用采集设备。probe 会逐个打开设备，设置请求模式，采样 `grab/retrieve/frame_interval`，并回读实际后端、分辨率、FPS、FourCC。

```bash
uv run python scripts/capture_probe.py \
  --device 0 \
  --formats MJPEG,YUY2,UYVY,RGB3,BGR3 \
  --fps 60,120,240 \
  --resolutions 1920x1080 \
  --samples 180 \
  --warmup 20 \
  --backend auto \
  --out-jsonl latency_runs/capture_probe_v1.2.0.jsonl \
  --out-md latency_runs/capture_probe_v1.2.0.md
```

判断时优先看实际 `fourcc/fps` 是否符合请求，再比较 `avg_frame_interval_ms`、`avg_grab_ms`、`avg_retrieve_ms` 和失败计数。

## 版本目录

当前 V1 实现在 `src/macos_dualbox_aim/v1/`。`scripts/main_v1.py` 显式导入
V1 版本路径，避免后续 V2 实验影响 V1 主链路。测试也应显式导入对应
major 版本路径。

`src/macos_dualbox_aim/core/` 只放跨 major 版本确认稳定的契约和通用代码。现在先保持
为空，避免把仍在变化的 `inference.py`、`kmbox.py`、控制器或 tuner 提前公共化。

包根目录只保留包入口和版本目录，不再放 `config.py`、`controller.py`、
`kmbox.py` 这类行为模块或兼容壳。新代码统一使用
`macos_dualbox_aim.v1.*`、`macos_dualbox_aim.v2.*` 这种显式版本路径。
