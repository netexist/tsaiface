# TSAIFACE · AI 视频换脸工具（中文版 · 开源）

> 本项目基于开源项目 **[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)**（作者 hacksider，上游版本 v2.1.6）二次开发，遵循 **AGPL-3.0** 开源协议。
>
> 本仓库提供的是**纯本地、去授权、去收费**的开源版本：原软件中的授权校验、试用限制、购买入口已全部移除，拿到源码即可直接运行，不需要任何激活。

## 功能

- **图片换脸**：单张图片替换人脸
- **视频换脸**：整段视频逐帧替换
- **实时换脸**：摄像头实时换脸，可用于直播 / 视频通话
- **多人脸对应**：一张图里多个人脸时，可手动指定谁换谁
- **口部遮罩**：减少嘴型不匹配的违和感
- **中文界面**：多语言文案在 `locales/` 下，可自行增改
- **深色 / 浅色皮肤**：左侧导航「切换皮肤」一键切换
- **三种运行方式**：CUDA（NVIDIA 显卡，最快）/ DirectML（AMD、Intel 显卡）/ CPU

## 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 / 11 64 位 |
| Python | 3.10 ~ 3.12 |
| ffmpeg | 需在 PATH 中，或把 `ffmpeg.exe`/`ffprobe.exe` 放在项目根目录 |
| 显卡 | 可选。NVIDIA 显卡走 CUDA 最快；没有独显用 CPU 也能跑，只是慢 |

## 安装

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 模型（不随仓库提供，需自行下载）

| 文件 | 放置位置 | 说明 |
|---|---|---|
| `inswapper_128_fp16.onnx` | `models/` | 换脸主模型（必需） |
| `gfpgan-1024.onnx` | `models/` | 人脸增强，可选 |
| `buffalo_l`（insightface 模型目录） | `models/buffalo_l/` | 人脸检测（必需） |

> ⚠️ 模型各有自己的许可：`inswapper_128` 官方仅授权**非商业用途**，商用前请自行确认授权。本仓库不包含任何模型权重。

## 运行

```bat
python run.py
```

或直接双击：

- `run-cuda.bat` —— NVIDIA 显卡
- `run-directml.bat` —— AMD / Intel 显卡

## 目录结构

```
modules/        核心代码（换脸、人脸分析、UI、摄像头等）
  processors/   逐帧处理器（换脸、增强、遮罩）
locales/        多语言文案
tests/          单元测试
assets/         图标
third_party/    上游原版源码（用于对照来源与差异）
run.py          程序入口
requirements.txt
```

## 与原版的差异（AGPL-3.0 要求明示修改内容）

1. **移除全部授权与收费逻辑**：删除 `modules/license.py`、`modules/ui_license.py`，并清掉界面上的「授权状态」入口、左下角授权状态文字、开始换脸 / 实时换脸前的授权拦截、后台心跳线程；
2. **界面中文化**，并增加深色 / 浅色皮肤切换；
3. 打包脚本、发布产物、证书与安装包相关文件不在本仓库内。

核心换脸算法与上游一致。上游原版源码归档见 `third_party/`，可用于逐文件对照。

## 免责声明

本项目仅供**学习研究**与**合法的内容创作**使用。

严禁用于伪造他人身份、诈骗、诽谤、色情、政治宣传等任何违法用途，也不得用于未经当事人同意的人脸替换。使用本项目产生的一切后果与法律责任由使用者自行承担，作者不承担任何责任。

## 许可证

本项目遵循 **GNU Affero General Public License v3.0（AGPL-3.0）**，全文见 [LICENSE](LICENSE)。

按 AGPL-3.0 要求，任何分发本软件（含修改版）的行为都必须同时提供完整源代码。
