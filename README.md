# 基于 YOLO 的工地安全生产智能监控系统

这是一个用于课程教学和工程原型验证的 PPE 监控系统。系统的输出统一称为“疑似违规”，不是对现场安全违规的最终认定；模型误报、漏报和摄像头偏差都需要人工复核。

## 快速开始

项目使用用户提供的 Python 解释器：

```powershell
D:\Program Files\Anaconda\envs\py310_env\python.exe -m pip install -r requirements.txt
D:\Program Files\Anaconda\envs\py310_env\python.exe scripts\download_datasets.py --dataset construction_ppe
D:\Program Files\Anaconda\envs\py310_env\python.exe scripts\convert_dataset.py
D:\Program Files\Anaconda\envs\py310_env\python.exe scripts\validate_dataset.py
D:\Program Files\Anaconda\envs\py310_env\python.exe scripts\train_model.py --smoke-test --confirm
D:\Program Files\Anaconda\envs\py310_env\python.exe -m streamlit run web\app.py
```

默认训练脚本不会替用户启动正式训练。完成数据校验并确认数据、模型、epochs、device 后，再运行：

```powershell
D:\Program Files\Anaconda\envs\py310_env\python.exe scripts\train_model.py --epochs 100 --device 0 --confirm
```

如果官方预训练权重或数据集下载不成功，脚本会保留可复现的下载 URL、错误日志和检查清单，不会删除 `datasets/raw` 中的原始文件。

如需补充安全帽/未戴安全帽样本，可先将 Voxel51 hard-hat-detection 图片和 `samples.json` 放入
`datasets/raw/voxel51_hardhat`，将 SHWD 官方 `VOC2028.zip` 放入 `datasets/raw/shwd`，再运行：

```powershell
& "D:\Program Files\Anaconda\envs\py310_env\python.exe" scripts\prepare_augmented_dataset.py
```

该脚本会保留原 Construction-PPE 数据，统一转换 `person/helmet/vest/no_helmet`，并生成
`datasets/processed_augmented/data.yaml`。SHWD 的 `hat/person` 和 Voxel51 的 `helmet/head` 映射会写入
`datasets/reports/augmented/conversion_summary.json`；没有明确标注的 vest 不会被反向伪造。

训练入口默认采用离线稳态配置：本地 `yolo11n.pt`、本地 `data.yaml`、`amp=false`、`plots=false`、`workers=0`。这样训练开始后断网不会触发 AMP 检查权重或字体下载，也能规避 Windows 多进程/OpenMP 冲突。

## 项目结构

```text
configs/       默认配置和训练参数
core/          路径、数据模型和安全边界
services/      数据集、训练、推理、规则、事件、报告和 LLM 服务
scripts/       下载、转换、校验、训练、评估、推理和报告命令
web/           Streamlit 可视化页面
tests/         不依赖模型权重的单元测试
datasets/      raw 原始数据、processed 统一 YOLO 数据和 reports
runs/          训练/评估输出
```

## 类别和数据使用原则

统一训练类别是 `person`、`helmet`、`vest`、`no_helmet`。`no_helmet` 只从原始数据明确标注了缺失安全帽的标签转换，不能因为模型没有检测到 `helmet` 就反向伪造这个标签。没有 PPE 标注的恶劣天气数据只用于鲁棒性测试或图像增强实验。

数据集来源和许可证会写入 `datasets/manifest.json`。使用前仍需按原始项目页面核对许可证、下载权限和引用要求。

## 安全边界

- 规则引擎只产生 `suspected_violation`，并写入 `needs_human_review` 状态。
- 不做人脸、身份、情绪、姿态或事故原因推断。
- API Key 只从环境变量读取，日志不会打印 Key。
- Agent 工具禁止删除原始数据、覆盖原始标注、自动发布告警或执行未确认的正式训练。
