# macos-dualbox-aim V1.2.3

V1.2.3 将主运行时回退到 V1.0.0 行为链路，只保留：

- 配置加载
- 实时 CoreML 推理入口
- 检测 bbox 坐标转换
- aim offset
- 四参 PIDF 控制器
- KMBox 热键监控与相对鼠标移动
- Web tuner 实时调参与保存配置

## V1.2.3 回退说明

V1.2.3 回退主程序采集实现：`RealtimeInference` 重新使用 V1.0 的 `capture.read()` 路径，不再在主循环中拆分 `grab/retrieve`，也不再把采集后端诊断塞进 tuner 快照。之前的独立诊断工具保留在脚本里，便于离线探索，不影响主程序运行。

主程序 tuner 里仍保留这些总链路延迟项：

- `capture_read_ms`
- `crop_ms`
- `queue_wait_ms`
- `preprocess_ms`
- `coreml_ms`
- `postprocess_ms`
- `inference_ms`
- `detection_callback_ms`
- `target_select_ms`
- `pid_ms`
- `kmbox_send_ack_ms`

独立工具仍支持：

- 采集后端、实际分辨率/FPS/FourCC/缓冲设置
- `scripts/capture_probe.py` 矩阵测试并输出 Markdown/JSONL
- `--load sleep|busy --load-ms N` 模拟消费节拍或同进程 CPU 调度压力
- `--load-placement inline|thread` 区分串行采集后负载和后台线程负载

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
  uv run python scripts/latency_tool.py capture --label v1.2.3 --run run$i --duration 60 --interval 0.5
done

uv run python scripts/latency_tool.py compare "latency_runs/*.jsonl" --baseline-label v1.0.0 --candidate-label v1.2.3 --out latency_runs/v1.0.0_vs_v1.2.3.md
```

对比报告里的主平均值使用 tuner 返回的 `latency.avg`，也就是每次抓到的 tuner 滚动平均值再做 run 间汇总；负数 delta/change 表示 candidate 更快。
跨版本对比时，先切到对应 tag/分支运行主程序并用匹配的 label 采集，再回到当前版本生成 compare 报告。

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
  --out-jsonl latency_runs/capture_probe_v1.2.2.jsonl \
  --out-md latency_runs/capture_probe_v1.2.2.md
```

判断时优先看实际 `fourcc/fps` 是否符合请求，再比较 `avg_frame_interval_ms`、`avg_grab_ms`、`avg_retrieve_ms` 和失败计数。

## 带负载采集 probe

用 `--load sleep --load-ms 9` 模拟每帧推理消费约 9ms 的节拍；用 `--load busy --load-ms 9` 粗略模拟同进程 CPU 调度压力。默认 `--load-placement inline` 保留 V1.2.1 行为，也就是采集一帧后串行执行负载。V1.2.2 新增 `--load-placement thread`，会在后台线程制造负载，同时主线程持续采集，并在 Markdown/JSONL 中输出 load 迭代数和实际 load 周期。二者都只影响 probe，不会修改主程序配置。

```bash
uv run python scripts/capture_probe.py \
  --device 0 \
  --formats MJPEG \
  --fps 120 \
  --resolutions 1920x1080 \
  --samples 180 \
  --warmup 20 \
  --load sleep \
  --load-ms 9 \
  --out-jsonl latency_runs/capture_probe_v1.2.2_sleep9_inline.jsonl \
  --out-md latency_runs/capture_probe_v1.2.2_sleep9_inline.md
```

推荐先跑下面这个 V1.2.2 区分实验：如果 `thread` + `busy9` 下 `interval` 仍接近 8.3ms，主链路采集退化就不像简单 CPU/GIL 竞争；如果退到 9.7-10ms，说明同进程负载竞争很可能是采集退化主因。

```bash
uv run python scripts/capture_probe.py \
  --device 0 \
  --formats MJPEG \
  --fps 120 \
  --resolutions 1920x1080 \
  --samples 180 \
  --warmup 20 \
  --load busy \
  --load-ms 9 \
  --load-placement thread \
  --out-jsonl latency_runs/capture_probe_v1.2.2_busy9_thread.jsonl \
  --out-md latency_runs/capture_probe_v1.2.2_busy9_thread.md
```

## 版本目录

当前 V1 实现在 `src/macos_dualbox_aim/v1/`。`scripts/main_v1.py` 显式导入
V1 版本路径，避免后续 V2 实验影响 V1 主链路。测试也应显式导入对应
major 版本路径。

`src/macos_dualbox_aim/core/` 只放跨 major 版本确认稳定的契约和通用代码。现在先保持
为空，避免把仍在变化的 `inference.py`、`kmbox.py`、控制器或 tuner 提前公共化。

包根目录只保留包入口和版本目录，不再放 `config.py`、`controller.py`、
`kmbox.py` 这类行为模块或兼容壳。新代码统一使用
`macos_dualbox_aim.v1.*`、`macos_dualbox_aim.v2.*` 这种显式版本路径。
