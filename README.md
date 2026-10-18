# YOLOv12 + MobileNetV3 安全帽检测 / Safety Helmet Detection

基于 YOLOv12 的安全帽佩戴检测与轻量化研究项目，提供 MobileNetV3 骨干网络改造、统一实验配置、模型评估，以及 ONNX / TensorRT 导出与部署工具。

A safety helmet detection and lightweight model research project built on YOLOv12, with MobileNetV3 backbone variants, shared experiment settings, evaluation, and ONNX / TensorRT deployment tools.

[中文说明](#中文说明) | [English](#english)

## 中文说明

### 项目特点

- 使用 SHWD 安全帽数据集，检测佩戴安全帽与未佩戴安全帽的头部。
- 提供 YOLOv12n、YOLOv12s 和三种 MobileNetV3 改造模型的对比流程。
- 保留多尺度检测结构，研究不同替换深度下的精度与计算成本。
- 支持数据检查、VOC 标注转换、训练、预测、评估和性能测试。
- 支持 PyTorch、ONNX 和 TensorRT 模型的部署推理与精度一致性验证。
- 自定义模块在 Python 进程内注册，无需修改已安装的 Ultralytics 源码。

### 模型方案

| 实验 | 模型 | 设计 |
|---|---|---|
| E0 | YOLOv12n | 小模型参照 |
| E1 | YOLOv12s | 基线模型 |
| E2 | MBV3-A | 替换最早的 C3k2 阶段 |
| E3 | MBV3-B | 替换 Attention 前的两个 C3k2 阶段 |
| E4 | MBV3-C | 在 B 的基础上替换后段下采样卷积，并压缩后段 Attention 通道 |

MobileNetV3 模块包含倒残差、深度卷积、SE 和 Hardswish。A/B/C 保留骨干后段 A2C2f Attention、多尺度 Neck 拓扑和 Detect Head；C 使用更窄的后段 Attention 通道。

本说明不宣称轻量化必然提高精度或推理速度。最终结论应以相同数据划分、训练设置和测试条件下的实测结果为准。

### 安装

使用 Python 3.10 或 3.11 创建虚拟环境。在项目根目录运行：

```sh
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```sh
source .venv/bin/activate
```

安装训练依赖：

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

需要 GPU 时，按目标运行环境安装匹配的 PyTorch 与 CUDA。具体依赖范围见 `requirements.txt`。

### 数据准备

数据需自行准备；运行示例前也需准备相应权重。将 YOLO 格式的数据放置为：

```text
data/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

类别映射与 `data/data.yaml` 一致：

| ID | 类别 | 含义 |
|---|---|---|
| 0 | `hat` | 佩戴安全帽的头部 |
| 1 | `person` | 未佩戴安全帽的头部，沿用数据集命名，并非完整人体框 |

每个标签文件的一行格式为 `class_id center_x center_y width height`，坐标归一化至 `[0, 1]`。如调整数据位置，请同步修改 `data/data.yaml`。

原始 VOC 数据可放入 `data/VOC2028/`，包含 `Annotations/`、`JPEGImages/` 和 `ImageSets/Main/`，并提供 `train.txt`、`val.txt` 划分，然后运行转换和检查：

```sh
python voc2yolo.py
python data_check.py --strict
```

已有 YOLO 格式数据时，仅运行检查命令。

### 训练与模型对比

训练前检查本地模型结构：

```sh
python smoke_test_models.py
```

默认实验配置位于 `configs/experiments.yaml`：80 个训练轮次、640 输入尺寸、batch 16、AdamW、随机种子 0；启用早停，实际训练可能提前结束。

```sh
# YOLOv12s 基线
python train.py

# MobileNetV3 改造模型
python train_mbv3.py --variant a
python train_mbv3.py --variant b
python train_mbv3.py --variant c
```

改造模型默认迁移形状匹配的 YOLOv12s 预训练参数，新层使用默认初始化。添加 `--no-pretrained` 可让改造模型从随机初始化开始训练。预训练文件需可在本地找到，或允许框架下载。

训练脚本支持 `--epochs`、`--batch`、`--imgsz`、`--device`、`--workers` 等参数。例如：

```sh
python train_mbv3.py --variant b --device cpu --batch 2 --workers 0
```

顺序训练并比较 E0–E4，或汇总已有标准路径中的权重：

```sh
python compare_models.py --train --device 0
python compare_models.py --device 0
```

`--device 0` 表示使用可用的第一块 CUDA GPU；支持 CPU 的脚本可改用 `--device cpu`。模型权重默认保存至 `runs/ablation/<实验名称>/weights/best.pt`。对比输出为 `runs/comparison/model_comparison.csv` 和 `model_comparison.png`，包括精度、模型复杂度和性能指标。

### 预测、评估与测速

以下示例使用 E3 训练得到的权重。请将 `sample.jpg` 换成实际输入，并在使用其他实验时同步替换权重路径。

```sh
python predict.py sample.jpg --weights runs/ablation/e3_mbv3_b/weights/best.pt --device 0
python evaluate.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0
python benchmark.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --warmup 50 --iterations 300 --output runs/comparison/mbv3_b_benchmark.json
```

预测会保存可视化和 YOLO TXT 结果，并统计两个类别的检测数。`benchmark.py` 测量原始模型前向时间，不包含读文件、预处理或 NMS；比较时应保持输入、batch、精度模式、预热和迭代次数一致。

### ONNX / TensorRT 部署

可选部署依赖位于 `requirements-deploy.txt`。该文件指定 CUDA 12 系列的 GPU 后端包，使用前应确认与目标环境兼容；TensorRT 导出需要兼容的 NVIDIA GPU 与 CUDA 环境。

```sh
python -m pip install -r requirements-deploy.txt
python export_onnx.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --imgsz 640 --batch 1 --opset 17 --output runs/deployment/mbv3_b.onnx
python export_tensorrt.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --imgsz 640 --batch 1 --fp16 --output runs/deployment/mbv3_b.engine
```

统一部署推理支持 `.pt`、`.onnx` 和 `.engine`：

```sh
python deploy_predict.py runs/deployment/mbv3_b.onnx sample.jpg --device 0 --conf 0.5 --save-txt
python deploy_predict.py runs/deployment/mbv3_b.engine 0 --device 0 --show --no-save
```

对比导出前后的精度和部署延迟：

```sh
python validate_deployment.py runs/ablation/e3_mbv3_b/weights/best.pt runs/deployment/mbv3_b.onnx runs/deployment/mbv3_b.engine --data data/data.yaml --device 0 --batch 1 --output runs/deployment/validation.csv
python benchmark_deployment.py runs/ablation/e3_mbv3_b/weights/best.pt runs/deployment/mbv3_b.onnx runs/deployment/mbv3_b.engine --device 0 --imgsz 640 --warmup 50 --iterations 300 --output runs/deployment/latency.csv
```

部署测速包含预处理、后端推理、NMS 和 Python 调用开销。TensorRT Engine 应在目标部署环境中构建。加载自定义 PyTorch 检查点时，请使用项目脚本，或先调用 `mobile_yolo.registry.register_ultralytics_modules()`。

## English

### Overview

This project studies safety helmet detection on SHWD using YOLOv12 and lightweight MobileNetV3 backbone variants. It includes data validation, VOC conversion, training, prediction, evaluation, benchmarking, and ONNX / TensorRT deployment.

Custom layers are registered within the Python process without modifying the installed Ultralytics source code.

### Model variants

| Experiment | Model | Design |
|---|---|---|
| E0 | YOLOv12n | Small-model reference |
| E1 | YOLOv12s | Baseline |
| E2 | MBV3-A | Replace the earliest C3k2 stage |
| E3 | MBV3-B | Replace two C3k2 stages before attention |
| E4 | MBV3-C | Extend B with later downsampling replacements and narrower attention channels |

The MobileNetV3 modules use inverted residuals, depthwise convolution, squeeze-and-excitation, and Hardswish. All three variants retain later backbone A2C2f attention, the multiscale neck topology, and the detection head. Variant C reduces later attention channel widths.

Lower computational complexity does not guarantee higher accuracy or lower latency. Compare results using the same data split, training settings, and benchmarking conditions.

### Installation

Create a Python 3.10 or 3.11 virtual environment from the project root:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux / macOS. Then install dependencies:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with the target CUDA environment. Dependency ranges are defined in `requirements.txt`.

### Dataset

Prepare the dataset separately. Place images under `data/images/train/` and `data/images/val/`, and matching YOLO labels under `data/labels/train/` and `data/labels/val/`. Update `data/data.yaml` if the locations change.

| ID | Class | Meaning |
|---|---|---|
| 0 | `hat` | Head wearing a safety helmet |
| 1 | `person` | Uncovered head; the original dataset name does not denote a full-body box |

Each label row is `class_id center_x center_y width height`, with normalized coordinates in `[0, 1]`.

For VOC conversion, place `Annotations/`, `JPEGImages/`, and `ImageSets/Main/` under `data/VOC2028/`, including `train.txt` and `val.txt` split files. Run:

```sh
python voc2yolo.py
python data_check.py --strict
```

Skip conversion if the dataset is already in YOLO format.

### Training and comparison

Check model construction before training:

```sh
python smoke_test_models.py
python train.py
python train_mbv3.py --variant a
python train_mbv3.py --variant b
python train_mbv3.py --variant c
```

Shared settings in `configs/experiments.yaml` use 80 epochs, image size 640, batch size 16, AdamW, and seed 0. Early stopping may end training before the configured epoch limit.

MobileNetV3 variants transfer compatible YOLOv12s pretrained parameters by default; new layers use default initialization. Use `--no-pretrained` for random initialization of these variants. Pretrained weights must be available locally or downloadable by the framework.

Training accepts overrides such as `--epochs`, `--batch`, `--imgsz`, `--device`, and `--workers`. For example:

```sh
python train_mbv3.py --variant b --device cpu --batch 2 --workers 0
```

Run all five experiments sequentially, or compare existing checkpoints at their standard locations:

```sh
python compare_models.py --train --device 0
python compare_models.py --device 0
```

`--device 0` selects the first available CUDA GPU. Use `--device cpu` with scripts that support CPU execution. Default checkpoints are saved to `runs/ablation/<experiment_name>/weights/best.pt`. Comparison reports are written to `runs/comparison/model_comparison.csv` and `model_comparison.png`.

### Prediction, evaluation, and benchmarking

These examples use the E3 checkpoint. Replace `sample.jpg` with an actual input and adjust the checkpoint path when using another experiment.

```sh
python predict.py sample.jpg --weights runs/ablation/e3_mbv3_b/weights/best.pt --device 0
python evaluate.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0
python benchmark.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --warmup 50 --iterations 300 --output runs/comparison/mbv3_b_benchmark.json
```

Prediction saves annotated outputs and YOLO TXT labels and reports counts for both classes. The PyTorch benchmark measures raw model forward time, excluding file loading, preprocessing, and NMS. Keep input size, batch size, precision, warmup, and iteration count consistent across comparisons.

### Deployment

Optional dependencies in `requirements-deploy.txt` target CUDA 12 GPU backends. Check compatibility with the deployment environment before installing. TensorRT export requires a compatible NVIDIA GPU and CUDA environment.

```sh
python -m pip install -r requirements-deploy.txt
python export_onnx.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --imgsz 640 --batch 1 --opset 17 --output runs/deployment/mbv3_b.onnx
python export_tensorrt.py runs/ablation/e3_mbv3_b/weights/best.pt --device 0 --imgsz 640 --batch 1 --fp16 --output runs/deployment/mbv3_b.engine
python deploy_predict.py runs/deployment/mbv3_b.onnx sample.jpg --device 0 --conf 0.5 --save-txt
python deploy_predict.py runs/deployment/mbv3_b.engine 0 --device 0 --show --no-save
```

Validate exported accuracy and measure deployment latency:

```sh
python validate_deployment.py runs/ablation/e3_mbv3_b/weights/best.pt runs/deployment/mbv3_b.onnx runs/deployment/mbv3_b.engine --data data/data.yaml --device 0 --batch 1 --output runs/deployment/validation.csv
python benchmark_deployment.py runs/ablation/e3_mbv3_b/weights/best.pt runs/deployment/mbv3_b.onnx runs/deployment/mbv3_b.engine --device 0 --imgsz 640 --warmup 50 --iterations 300 --output runs/deployment/latency.csv
```

Deployment latency includes preprocessing, backend inference, NMS, and Python call overhead. Build TensorRT engines in the target deployment environment. Load custom PyTorch checkpoints through the project scripts, or call `mobile_yolo.registry.register_ultralytics_modules()` first.

## 项目结构 / Project structure

```text
.
├── configs/experiments.yaml       # 统一实验配置 / Shared experiment settings
├── data/data.yaml                 # 数据配置 / Dataset configuration
├── mobile_yolo/                   # 自定义模块与运行支持 / Custom modules and runtime
├── models/                        # 模型结构 / Model architectures
├── requirements.txt              # 训练依赖 / Training dependencies
├── requirements-deploy.txt        # 部署依赖 / Deployment dependencies
├── voc2yolo.py                    # VOC 转换 / VOC conversion
├── data_check.py                  # 数据检查 / Dataset checks
├── smoke_test_models.py           # 结构检查 / Architecture checks
├── train.py                      # 基线训练 / Baseline training
├── train_mbv3.py                  # 改造模型训练 / Variant training
├── compare_models.py              # 实验对比 / Experiment comparison
├── predict.py                     # 预测 / Prediction
├── evaluate.py                    # 评估 / Evaluation
├── benchmark.py                   # 模型前向测速 / Model forward benchmark
├── export_onnx.py                 # ONNX 导出 / ONNX export
├── export_tensorrt.py              # TensorRT 导出 / TensorRT export
├── deploy_predict.py              # 部署推理 / Deployment inference
├── validate_deployment.py         # 部署精度验证 / Deployment accuracy validation
└── benchmark_deployment.py        # 部署测速 / Deployment benchmark
```
