"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

# 窗口默认尺寸，同时也是最小尺寸：再小就会把「处理选项」「细节微调」两块
# 挤压到看不清（实测 850×900 各区域刚好完整显示）。可放大，不能缩小。
ROOT_HEIGHT = 900
ROOT_WIDTH = 850

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS_DARK = """
/* ===== TSAIFACE · TSRVC 风格深色主题 =====
   配色来源：TSRVC 项目 web/app.css 的深色主题变量
   背景 #0F1115 / 卡片 #171A20 / 次级 #1E222A / 描边 #262B33 #323844
   主色（品牌绿）#22C55E  危险红 #EF4444  文字 #E6E9EE #8B95A3 #5C6675
*/
QMainWindow, QDialog { background-color: #0F1115; }
QWidget {
    color: #E6E9EE;
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", Arial, sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #171A20;
    border: 1px solid #262B33;
    border-radius: 14px;
    margin-top: 0;
    /* 顶部留出空间给标题：标题画在框内，而不是跑到框外面去 */
    padding: 38px 14px 14px 14px;
    font-weight: 600;
}
QGroupBox::title {
    /* subcontrol-origin 默认 padding：以边框内侧为基准，标题就落在框里 */
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 16px;
    top: 12px;
    color: #22C55E;
}

QPushButton {
    background-color: #22C55E;
    color: #08170D;
    border: none;
    border-radius: 10px;
    padding: 9px 18px;
    font-weight: 700;
}
QPushButton:hover  { background-color: #2ED573; }
QPushButton:pressed{ background-color: #16A34A; }
QPushButton:disabled { background-color: #262B33; color: #5C6675; }
QPushButton#secondary {
    background-color: #1E222A;
    color: #E6E9EE;
    border: 1px solid #323844;
}
QPushButton#secondary:hover { background-color: #262B33; border-color: #22C55E; }
QPushButton#danger { background-color: #EF4444; color: #ffffff; }
QPushButton#danger:hover  { background-color: #F87171; }

QComboBox {
    background-color: #1E222A;
    border: 1px solid #323844;
    border-radius: 10px;
    padding: 7px 12px;
    min-height: 26px;
    color: #E6E9EE;
}
QComboBox:hover { border-color: #22C55E; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #1E222A;
    color: #E6E9EE;
    selection-background-color: rgba(34,197,94,0.16);
    selection-color: #22C55E;
    border: 1px solid #323844;
    outline: none;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QLineEdit {
    background-color: #1E222A;
    border: 1px solid #323844;
    border-radius: 10px;
    padding: 7px 12px;
    color: #E6E9EE;
    selection-background-color: #22C55E;
    selection-color: #08170D;
}
QLineEdit:focus { border-color: #22C55E; }
QLineEdit[readOnly="true"] { color: #8B95A3; background-color: #171A20; }
QCheckBox::indicator {
    width: 38px; height: 20px;
    border-radius: 10px;
    background-color: #323844;
}
QCheckBox::indicator:checked {
    background-color: #22C55E;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #262B33;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #E6E9EE;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid #0F1115;
}
QSlider::sub-page:horizontal {
    background: #22C55E;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #171A20;
    border: 2px dashed #323844;
    border-radius: 14px;
    color: #8B95A3;
}
/* 可点击的图片框：鼠标移上去虚线变亮绿，提示"这里能点" */
QLabel#imageDrop:hover {
    border-color: #22C55E;
    color: #E6E9EE;
}
QLabel#statusLabel {
    color: #8B95A3;
    font-size: 9.5pt;
}
/* 说明卡片（实时换脸页的使用指引），独立命名便于深浅主题分别配色 */
QLabel#hintCard {
    color: #8B95A3;
    background-color: #171A20;
    border: 1px solid #262B33;
    border-radius: 14px;
    padding: 14px;
}
QLabel#linkLabel {
    color: #22C55E;
    text-decoration: none;
}
QLabel#brandTitle {
    color: #22C55E;
    font-size: 15pt;
    font-weight: 800;
}
QLabel#brandSub {
    color: #5C6675;
    font-size: 8.5pt;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #171A20;
    border: 1px solid #262B33;
    border-radius: 14px;
}

/* ===== 左侧导航栏（参考 TSRVC 的 .side）===== */
QWidget#sideNav {
    background-color: #171A20;
    border-right: 1px solid #262B33;
}
QWidget#mainArea { background-color: #0F1115; }
QPushButton#navItem {
    background-color: transparent;
    color: #8B95A3;
    border: none;
    border-radius: 10px;
    text-align: left;
    padding: 8px 12px;
    font-weight: 500;
}
QPushButton#navItem:hover {
    background-color: #1E222A;
    color: #E6E9EE;
}
QPushButton#navItem[active="true"] {
    background-color: rgba(34,197,94,0.14);
    color: #22C55E;
    font-weight: 700;
}

/* ===== 底部进度条 ===== */
QProgressBar {
    background-color: #1E222A;
    border: 1px solid #262B33;
    border-radius: 10px;
    text-align: center;
    color: #E6E9EE;
    font-size: 10.5pt;
}
QProgressBar::chunk {
    background-color: #22C55E;
    border-radius: 9px;
}
"""

# ===== 浅色主题 =====
# 结构与深色主题一一对应，只换配色，保证切换后没有任何元素"漏色"变成深色块。
# 浅底上绿色文字改用更深的 #16A34A，避免 #22C55E 在白底上发飘、对比度不足。
QSS_LIGHT = """
QMainWindow, QDialog { background-color: #F4F6FA; }
QWidget {
    color: #1A1F2B;
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", Arial, sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #DFE4EC;
    border-radius: 14px;
    margin-top: 0;
    padding: 38px 14px 14px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 16px;
    top: 12px;
    color: #16A34A;
}

QPushButton {
    background-color: #16A34A;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 9px 18px;
    font-weight: 700;
}
QPushButton:hover  { background-color: #22C55E; }
QPushButton:pressed{ background-color: #15803D; }
QPushButton:disabled { background-color: #E7EBF2; color: #9AA4B2; }
QPushButton#secondary {
    background-color: #F1F4F9;
    color: #1A1F2B;
    border: 1px solid #D9DEE8;
}
QPushButton#secondary:hover { background-color: #E7ECF4; border-color: #16A34A; }
QPushButton#danger { background-color: #DC2626; color: #FFFFFF; }
QPushButton#danger:hover  { background-color: #EF4444; }

QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D9DEE8;
    border-radius: 10px;
    padding: 7px 12px;
    min-height: 26px;
    color: #1A1F2B;
}
QComboBox:hover { border-color: #16A34A; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #1A1F2B;
    selection-background-color: rgba(22,163,74,0.14);
    selection-color: #16A34A;
    border: 1px solid #D9DEE8;
    outline: none;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D9DEE8;
    border-radius: 10px;
    padding: 7px 12px;
    color: #1A1F2B;
    selection-background-color: #16A34A;
    selection-color: #FFFFFF;
}
QLineEdit:focus { border-color: #16A34A; }
QLineEdit[readOnly="true"] { color: #5C6675; background-color: #F4F6FA; }
QCheckBox::indicator {
    width: 38px; height: 20px;
    border-radius: 10px;
    background-color: #D9DEE8;
}
QCheckBox::indicator:checked {
    background-color: #16A34A;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #E3E8EF;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid #16A34A;
}
QSlider::sub-page:horizontal {
    background: #22C55E;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 14px;
    color: #5C6675;
}
QLabel#imageDrop:hover {
    border-color: #16A34A;
    color: #1A1F2B;
}
QLabel#statusLabel {
    color: #5C6675;
    font-size: 9.5pt;
}
QLabel#hintCard {
    color: #5C6675;
    background-color: #FFFFFF;
    border: 1px solid #DFE4EC;
    border-radius: 14px;
    padding: 14px;
}
QLabel#linkLabel {
    color: #16A34A;
    text-decoration: none;
}
QLabel#brandTitle {
    color: #16A34A;
    font-size: 15pt;
    font-weight: 800;
}
QLabel#brandSub {
    color: #8B95A3;
    font-size: 8.5pt;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #DFE4EC;
    border-radius: 14px;
}

QWidget#sideNav {
    background-color: #FFFFFF;
    border-right: 1px solid #DFE4EC;
}
QWidget#mainArea { background-color: #F4F6FA; }
QPushButton#navItem {
    background-color: transparent;
    color: #5C6675;
    border: none;
    border-radius: 10px;
    text-align: left;
    padding: 8px 12px;
    font-weight: 500;
}
QPushButton#navItem:hover {
    background-color: #F1F4F9;
    color: #1A1F2B;
}
QPushButton#navItem[active="true"] {
    background-color: rgba(22,163,74,0.12);
    color: #15803D;
    font-weight: 700;
}

QProgressBar {
    background-color: #EEF2F7;
    border: 1px solid #DFE4EC;
    border-radius: 10px;
    text-align: center;
    color: #1A1F2B;
    font-size: 10.5pt;
}
QProgressBar::chunk {
    background-color: #22C55E;
    border-radius: 9px;
}
"""

# 兼容旧引用（探针脚本、init 默认值都用 QSS）
QSS = QSS_DARK

# ─── 主题（深色 / 浅色）────────────────────────────────────────────────────

# 当前主题：dark / light。由 init() 从 config/theme.txt 读取后应用
_THEME = "dark"


def _theme_file_path() -> str:
    """主题设置文件位置（项目内 config/theme.txt，与用户数据一起不进仓库）。"""
    import modules.paths as _paths

    return os.path.join(_paths.ROOT_DIR, "config", "theme.txt")


def load_theme() -> str:
    """读取上次选择的主题；读不到或内容非法时回退深色。"""
    try:
        with open(_theme_file_path(), "r", encoding="utf-8") as fh:
            name = fh.read().strip().lower()
        if name in ("dark", "light"):
            return name
    except Exception:
        pass
    return "dark"


def save_theme(name: str) -> None:
    """保存主题选择，下次启动沿用。写失败不影响当前运行。"""
    try:
        p = _theme_file_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(name)
    except Exception:
        pass


def apply_theme(name: str) -> None:
    """把主题应用到整个应用（所有窗口共用 QSS，切换后立即生效）。"""
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(QSS_LIGHT if name == "light" else QSS_DARK)


def current_theme() -> str:
    """供其它模块按当前主题选配色。"""
    return _THEME


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
# How long to wait quietly for a live-preview worker to notice the stop flag
# before saying so. Shutdown then keeps waiting: it has to finish.
WORKER_SHUTDOWN_GRACE_MS = 30000
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "图片 (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "图片与视频 (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "视频 (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)
    progressChanged = Signal(int, int)  # (已处理数, 总数)：视频处理进度


def set_progress(current: int, total: int) -> None:
    """供 core 在处理视频时上报进度。

    特点：
      · 跨线程安全（走 Qt 信号）；
      · 处理本身是在 UI 线程同步跑的，这里主动 pump 一次事件循环，
        否则界面要等到全部处理完才会重绘，用户会以为程序卡死。
    """
    if _BRIDGE is not None:
        _BRIDGE.progressChanged.emit(int(current), int(total))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        _APP.processEvents()


# 当前正在执行的任务（取最近一条状态文字），进度条上会一起显示
_STAGE_TEXT = "处理中"


def _emit_status(text: str) -> None:
    global _STAGE_TEXT
    _STAGE_TEXT = text
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("已跳过处理")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["未检测到摄像头"]
        except Exception as exc:
            print(f"检测摄像头出错：{exc}")
            return [], ["未检测到摄像头"]

    if platform.system() == "Darwin":
        return [0, 1], ["摄像头 0", "摄像头 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"摄像头 {i}")
            cap.release()
    return (indices, names) if names else ([], ["未检测到摄像头"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


def _make_clickable(label: QLabel, callback: Callable[[], None], tip: str = "") -> None:
    """把一个静态 QLabel 变成可点击区域。

    为什么需要：源人脸 / 目标画面两个虚线框只是"预览占位"，用户很自然会
    直接点它，但 QLabel 没有 clicked 信号。这里接管鼠标按下事件，
    效果与点下面的按钮完全一致；再次点击等于更换内容。
    """
    label.setCursor(Qt.CursorShape.PointingHandCursor)
    if tip:
        label.setToolTip(tip)

    def _on_press(event):
        # 只响应左键，右键 / 中键不误触发文件选择框
        if event.button() == Qt.MouseButton.LeftButton:
            callback()

    label.mousePressEvent = _on_press


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb
        self._current_mode = "image"   # image / video，决定目标文件选择的过滤器
        # 底部状态区状态机：idle（只显示文字）/ running（只显示进度条）/ done（进度条满格展示结果）
        self._proc_state = "idle"

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
        )
        # 显式给主窗口设置图标：app.setWindowIcon 在部分 Windows 环境下
        # 不会传递给之后创建的窗口，这里再设一次，保证左上角与 Alt+Tab 都正确。
        _icon_path = _find_app_icon()
        if _icon_path:
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(_icon_path))
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        # 整体：左侧导航（196px）+ 右侧主工作区
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        shell.addWidget(self._build_left_nav())

        right = QWidget()
        right.setObjectName("mainArea")
        layout = QVBoxLayout(right)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)
        shell.addWidget(right, 1)

        # ── 顶部品牌栏（左侧品牌，右侧运行环境 + 显卡）──
        # 品牌名已在左侧导航显示，这里只保留一个轻量的页头（避免重复显示 TSAIFACE）
        layout.addWidget(self._build_brand_bar())

        # ── 中间内容区：工作台 / 实时换脸 两页切换 ──
        self._stack = QStackedWidget()
        self._page_work = self._build_work_page()
        self._page_live = self._build_live_page()
        self._stack.addWidget(self._page_work)
        self._stack.addWidget(self._page_live)
        layout.addWidget(self._stack, 1)

        # ── 底部：状态文字 + 进度条 ──
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(6)

        self._status_label = QLabel("状态：就绪，等待选择人脸照片")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_lay.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        # 高度与「开始换脸」等按钮一致，视觉上是一条完整的操作条，不是细缝
        self._progress.setFixedHeight(40)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        bottom_lay.addWidget(self._progress)
        layout.addWidget(bottom)

        # 页脚：AGPL-3.0 开源声明
        footer = QLabel("基于 Deep-Live-Cam 开源项目（AGPL-3.0）· 中文整合版")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)

    # ── 左侧导航 ──────────────────────────────────────────────────────────

    def _build_left_nav(self) -> QWidget:
        """左侧导航栏（参考 TSRVC 的 .side：品牌区 + 导航项 + 底部版本）。"""
        side = QWidget()
        side.setObjectName("sideNav")
        side.setFixedWidth(196)
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 18, 12, 14)
        v.setSpacing(4)

        # 品牌区
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("TSAIFACE")
        brand.setObjectName("brandTitle")
        sub = QLabel("AI FACE SWAP")
        sub.setObjectName("brandSub")
        brand_box.addWidget(brand)
        brand_box.addWidget(sub)
        wrap = QWidget()
        wrap.setLayout(brand_box)
        wrap.setContentsMargins(8, 0, 8, 18)
        v.addWidget(wrap)

        # 导航项
        self._nav_buttons = {}
        items = [
            ("image", "图片换脸", self._on_nav_image),
            ("video", "视频换脸", self._on_nav_video),
            ("live", "实时换脸", self._on_nav_live),
            ("theme", "切换皮肤", self._on_toggle_theme),
            ("about", "关于软件", self._on_show_about),
        ]
        for key, text, cb in items:
            btn = QPushButton(text)
            btn.setObjectName("navItem")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(cb)
            btn.setMinimumHeight(38)
            v.addWidget(btn)
            self._nav_buttons[key] = btn

        # 皮肤切换项：与其它导航项同款样式
        self._nav_theme = self._nav_buttons["theme"]
        self._nav_theme.setToolTip("在深色 / 浅色界面之间切换")

        v.addStretch(1)

        # 底部：版本
        self._nav_version = QLabel("v" + modules.metadata.version)
        self._nav_version.setObjectName("brandSub")
        v.addWidget(self._nav_version)

        self._set_nav_active("image")
        return side

    def _on_toggle_theme(self) -> None:
        global _THEME
        _THEME = "light" if _THEME == "dark" else "dark"
        apply_theme(_THEME)
        save_theme(_THEME)
        # 按钮文案固定为「切换皮肤」，用提示说明点了会发生什么
        self._nav_theme.setToolTip(
            "当前为浅色皮肤，点击切换到深色" if _THEME == "light"
            else "当前为深色皮肤，点击切换到浅色"
        )
        # 虚线框内的建议文字按主题取色，重新生成一次（已选图片的不重渲染）
        self._refresh_source_views(text_only=True)
        self._refresh_target_hint()
        update_status("已切换到%s皮肤" % ("浅色" if _THEME == "light" else "深色"))

    def _set_nav_active(self, key: str) -> None:
        for k, b in self._nav_buttons.items():
            b.setProperty("active", k == key)
            b.style().unpolish(b)
            b.style().polish(b)

    # ── 工作台页（源/目标 + 参数 + 开始按钮）────────────────────────────

    def _build_work_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addLayout(self._build_image_row())
        lay.addWidget(self._build_options_card())
        lay.addWidget(self._build_sliders_card())
        lay.addLayout(self._build_action_row())
        lay.addStretch(1)
        return page

    # ── 实时换脸页 ──────────────────────────────────────────────────────

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        lay.addWidget(self._build_live_source_card())
        lay.addWidget(self._build_camera_card())
        tip = QLabel(
            "实时换脸说明：\n"
            "  1. 先在上方「源人脸」里选好要换上去的人脸照片（点虚框或点按钮都可以）；\n"
            "  2. 选择摄像头与分辨率，点「实时换脸」；\n"
            "  3. 弹出的实时窗口会显示换脸效果，可全屏用于直播；\n"
            "  4. 分辨率越高越吃显卡，建议 NVIDIA 独立显卡，核显会非常卡。"
        )
        tip.setObjectName("hintCard")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        lay.addStretch(1)
        return page

    def _build_live_source_card(self) -> QGroupBox:
        """实时换脸页的源人脸选择区（整块水平居中）。

        为什么要单独做一块：原来必须先回「图片换脸」页选人脸再切回本页，
        来回点左侧导航很麻烦。这里复用同一个 source_path 与同一套选择逻辑，
        选完两边的预览框会同步更新。
        """
        card = QGroupBox(_("源人脸"))
        v = QVBoxLayout(card)
        v.setSpacing(10)

        self.live_source_label = _make_image_drop(_("源人脸"), (200, 200))
        _make_clickable(
            self.live_source_label, self._on_select_source,
            _("点击选择 / 更换源人脸照片"),
        )
        v.addWidget(self.live_source_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        row = QHBoxLayout()
        row.addStretch(1)
        self.live_btn_select_source = QPushButton(_("选择人脸照片"))
        self.live_btn_select_source.setToolTip(_("选择要换上去的人脸照片"))
        self.live_btn_select_source.clicked.connect(self._on_select_source)
        row.addWidget(self.live_btn_select_source)
        self.live_btn_random_face = QPushButton("随机")
        self.live_btn_random_face.setObjectName("secondary")
        self.live_btn_random_face.setFixedWidth(62)
        self.live_btn_random_face.setToolTip(_("随机获取一张人脸（需联网）"))
        self.live_btn_random_face.clicked.connect(self._on_random_face)
        row.addWidget(self.live_btn_random_face)
        row.addStretch(1)
        v.addLayout(row)

        # 框内占位文字同样带使用建议（复用同一套生成逻辑）
        self.live_source_label.setText(self._source_drop_text())
        return card

    # ── 源人脸预览同步 ───────────────────────────────────────────────────

    def _source_preview_labels(self) -> List[QLabel]:
        """所有显示「源人脸」预览的位置：图片/视频页 + 实时换脸页。"""
        labels = [self.source_label]
        live = getattr(self, "live_source_label", None)
        if live is not None:
            labels.append(live)
        return labels

    def _refresh_source_views(self, text_only: bool = False) -> None:
        """把当前 source_path 同步到所有源人脸预览框。

        两个页面各有自己的 QLabel，但共用同一个 modules.globals.source_path，
        所以选完 / 随机 / 交换之后必须两边一起刷新，否则会有一边显示旧图。
        渲染只做一次，QPixmap 隐式共享，多给几个 label 不会重复解码。

        text_only=True 用于主题切换：已有预览图时不重新解码，只更新占位文字。
        """
        path = modules.globals.source_path
        labels = self._source_preview_labels()
        if path and is_image(path):
            if text_only:
                return
            pm = render_image_preview(path, (200, 200))
            for lb in labels:
                lb.setPixmap(pm)
                lb.setText("")
        else:
            fill = self._source_drop_text()
            for lb in labels:
                lb.clear()
                lb.setText(fill)

    # ── 导航行为 ────────────────────────────────────────────────────────

    def _on_nav_image(self) -> None:
        self._current_mode = "image"
        self._set_nav_active("image")
        self._stack.setCurrentWidget(self._page_work)
        self._page_title.setText("图片换脸")
        self._refresh_target_hint()
        self.set_status("状态：请选择一张人脸照片和一张目标图片")

    def _on_nav_video(self) -> None:
        self._current_mode = "video"
        self._set_nav_active("video")
        self._stack.setCurrentWidget(self._page_work)
        self._page_title.setText("视频换脸")
        self._refresh_target_hint()
        self.set_status("状态：请选择一张人脸照片和一个目标视频")

    def _on_nav_live(self) -> None:
        self._set_nav_active("live")
        self._stack.setCurrentWidget(self._page_live)
        self._page_title.setText("实时换脸")
        if not (modules.globals.source_path and is_image(modules.globals.source_path)):
            self.set_status("状态：先选「源人脸」，再选摄像头，点「实时换脸」")
        else:
            self.set_status("状态：选择摄像头后点「实时换脸」")

    def _on_show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "关于 TSAIFACE",
            "<h3>TSAIFACE %s %s</h3>"
            "<p>一款本地运行的中文 AI 换脸工具。</p>"
            "<p>支持图片换脸、视频换脸、实时摄像头换脸。</p>"
            "<p style='color:#8B95A3'>基于开源项目 Deep-Live-Cam（AGPL-3.0）</p>"
            "<p style='color:#8B95A3'>源代码随安装包提供，可自由查看与修改。</p>"
            % (modules.metadata.version, modules.metadata.edition),
        )

    # ── 进度条（供 core 处理过程中回调）─────────────────────────────────

    def set_progress(self, current: int, total: int) -> None:
        """显示确定进度；total<=0 时退化为不确定进度（一直滚动）。

        进度条文字里同时显示「当前正在执行什么」，用户不用盯着底部状态栏，
        也能知道程序是在干活而不是卡死。
        """
        self._progress.setVisible(True)
        # 进度条接管底部状态区，文字状态行让位（避免同一件事显示两遍）
        self._status_label.hide()
        # Qt 的格式串里 % 是占位符，状态文字中若含 "100%" 之类必须先转义
        stage = _STAGE_TEXT.replace("%", "%%")[:30]
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(current, total))
            self._progress.setFormat(stage + "  %v/%m 帧（%p%）")
        else:
            self._progress.setRange(0, 0)
            self._progress.setFormat(stage)

    def hide_progress(self) -> None:
        self._progress.setVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        # 清掉残留文字，避免下次显示时一闪而过旧结果
        self._progress.setFormat("")


    def _build_brand_bar(self) -> QWidget:
        """顶部页头。

        只放「当前功能页标题 + 运行环境」。
        """
        bar = QFrame()
        bar.setObjectName("card")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 10, 16, 10)
        h.setSpacing(10)

        # 左：当前页标题
        self._page_title = QLabel("换脸工作台")
        self._page_title.setObjectName("brandTitle")
        h.addWidget(self._page_title)
        h.addStretch(1)

        # 右：运行环境状态（GPU 名称 / 运算引擎）
        self._gpu_label = QLabel(self._describe_accelerator())
        self._gpu_label.setObjectName("statusLabel")
        h.addWidget(self._gpu_label)
        return bar

    @staticmethod
    def _describe_accelerator() -> str:
        """生成运行环境说明文字：能拿到显卡名就显示显卡，否则显示运算引擎。"""
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
        except Exception:
            return "运算引擎：未知"
        if "CUDAExecutionProvider" in providers:
            name = "GPU 加速"
            try:
                import subprocess
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip().splitlines()
                if out:
                    name = out[0]
            except Exception:
                pass
            return f"运算引擎：{name}"
        return "运算引擎：CPU（未检测到 NVIDIA 显卡）"

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("源人脸"), (200, 200))
        # 点击虚线框 = 点「选择人脸照片」，已选过就是更换
        _make_clickable(
            self.source_label, self._on_select_source,
            _("点击选择 / 更换源人脸照片"),
        )
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("选择人脸照片"))
        self.btn_select_source.setToolTip(
            _("选择要换上去的人脸照片")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_random_face = QPushButton("随机")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(62)
        self.btn_random_face.setToolTip(
            _("随机获取一张人脸（需联网）")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("交换")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(60, 34)
        self.btn_swap.setToolTip(_("交换源人脸和目标图片/视频"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        # 占位文字随当前模式变化：图片页显示「目标图片」，视频页显示「目标视频」
        self.target_label = _make_image_drop(_("目标图片"), (200, 200))
        # 点击虚线框 = 点「选择目标文件」，已选过就是更换
        _make_clickable(
            self.target_label, self._on_select_target,
            _("点击选择 / 更换目标图片或视频"),
        )
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("选择目标文件"))
        self.btn_select_target.setToolTip(
            _("选择要换脸的目标图片或视频")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        # 初始占位文字（含使用建议），后续由 _refresh_source_views / _refresh_target_hint 维护
        self._refresh_source_views()
        self._refresh_target_hint()
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("处理选项"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "保持原帧率",
                                "输出视频保持原帧率")
        self.sw_keep_audio = make("keep_audio", "保留原音轨",
                                  "把原视频的音轨复制到输出文件")
        self.sw_keep_frames = make("keep_frames", "保留临时帧",
                                   "处理完保留提取出的帧文件")
        self.sw_many_faces = make("many_faces", "多人脸模式",
                                  "替换画面中的所有脸，不只最主要那张")
        self.sw_poisson = make("poisson_blend", "边缘融合",
                               "用泊松融合让换脸边缘更自然")
        self.sw_color_fix = make("color_correction", "修正摄像头偏色",
                                 "修正部分摄像头偏蓝/偏绿的问题")
        self.sw_show_fps = make("show_fps", "显示帧率",
                                "在实时预览上显示每秒帧数")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("多人脸指定"), modules.globals.map_faces,
                                    _("手动指定哪张源人脸对应目标里的人脸"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown
        enhancer_label = QLabel(_("美颜增强："))
        grid.addWidget(enhancer_label, len(items) // 2, 0)

        self.cb_enhancer = QComboBox()
        # 选项后面直接带一句人话解释，用户不用猜 GFPGAN / GPEN 分别是什么意思
        self._enhancer_items = [
            ("关闭", "不做美颜，只用换脸模型本身的效果"),
            ("GFPGAN（写实自然 · 推荐）", "老照片修复同款模型，质感真实、不假面，速度适中"),
            ("GPEN-512（最清晰 · 较慢）", "512 分辨率重建五官，细节最强，但每帧更耗时"),
            ("GPEN-256（速度快）", "256 分辨率重建，速度快、显存占用低，细节弱一些"),
        ]
        for label, tip in self._enhancer_items:
            self.cb_enhancer.addItem(label)
            self.cb_enhancer.setItemData(
                self.cb_enhancer.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        initial = "关闭"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN（写实自然 · 推荐）"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512（最清晰 · 较慢）"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256（速度快）"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("选择美颜模型（关闭 = 不做增强）"))
        grid.addWidget(self.cb_enhancer, len(items) // 2, 1)

        return card

    # ── 虚线框内的占位文字（标题 + 使用建议）────────────────────────────

    def _drop_placeholder(self, title: str, hint: str) -> str:
        """生成虚线框内的富文本占位：第一行标题，下面一行小字使用建议。

        为什么放框里而不是框外：框外文字会和下面的按钮挤在一起，
        容易被当成按钮的说明；放进框内，用户一眼就知道是在提示"这里放什么素材"。
        颜色按当前主题取，主题切换后会重新生成。
        """
        light = _THEME == "light"
        title_c = "#1A1F2B" if light else "#E6E9EE"
        hint_c = "#7A8494" if light else "#6F7A88"
        return (
            '<div style="text-align:center">'
            '<span style="font-size:12pt;color:%s">%s</span>'
            '<br><br>'
            '<span style="font-size:8pt;color:%s">%s</span>'
            '</div>' % (title_c, title, hint_c, hint)
        )

    def _source_drop_text(self) -> str:
        """源人脸框的占位内容（含建议）。"""
        return self._drop_placeholder(
            _("源人脸"),
            "建议：正脸、五官清晰<br>无遮挡、光线均匀",
        )

    # ── 目标占位文字 / 使用建议（随图片、视频模式切换）────────────────────

    def _target_drop_text(self) -> str:
        """目标虚线框的占位文字：图片页叫「目标图片」，视频页叫「目标视频」。"""
        return _("目标视频") if self._current_mode == "video" else _("目标图片")

    def _has_target_preview(self) -> bool:
        """目标框里是否已经放入了预览图（有图就不要覆盖它的文字）。"""
        try:
            pm = self.target_label.pixmap()
            return pm is not None and not pm.isNull()
        except Exception:
            return False

    def _refresh_target_hint(self) -> None:
        """同步目标框内的占位文字与建议。

        只在还没选文件时改文字，已选文件的预览图保持不变，
        避免切换功能页时把用户已经选好的目标"改回文字"。
        """
        if self._has_target_preview():
            return
        if self._current_mode == "video":
            hint = "建议：画面稳定、人脸清晰<br>分辨率不低于 720p，人脸别太小"
        else:
            hint = "建议：照片清晰、人脸占比较大<br>少遮挡、光线均匀"
        self.target_label.setText(
            self._drop_placeholder(self._target_drop_text(), hint)
        )

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("细节微调"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("融合度")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("原图与换脸结果的混合比例（0% = 原图，100% = 完全换脸）")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("锐化度")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("让美颜后的人脸更锐利"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("嘴部遮罩")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = 用换脸后的嘴部，100 = 露出原始嘴部到下巴")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("开始换脸"))
        self.btn_start.setToolTip(_("用选定的人脸开始处理目标图片/视频"))
        self.btn_start.clicked.connect(self._on_start)

        # 预览按钮：默认「预览」，预览窗口打开后同一按钮变成「关闭预览」（红底）
        self.btn_preview = QPushButton(_("预览"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("显示/隐藏处理结果的预览窗口"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        # 打开目录：跳到输出文件所在文件夹（转换前用目标文件目录，完成后用保存目录）
        self.btn_open_dir = QPushButton(_("打开目录"))
        self.btn_open_dir.setObjectName("secondary")
        self.btn_open_dir.setToolTip(_("在资源管理器中打开输出文件所在目录"))
        self.btn_open_dir.clicked.connect(self._on_open_dir)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_open_dir)
        # 三个按钮统一固定高度，和下方进度条保持一致的操作条观感
        for b in (self.btn_start, self.btn_preview, self.btn_open_dir):
            b.setFixedHeight(40)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("摄像头"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        # Row 0: camera selector + Live button
        grid.addWidget(QLabel(_("选择摄像头：")), 0, 0)
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "未检测到摄像头":
            self.cb_camera.addItem("未检测到摄像头")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("选择实时换脸使用的摄像头"))
        grid.addWidget(self.cb_camera, 0, 1)

        self.btn_live = QPushButton(_("实时换脸"))
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("用摄像头开始实时换脸"))
        self.btn_live.clicked.connect(self._on_live)
        grid.addWidget(self.btn_live, 0, 2)

        # Row 1: capture resolution
        grid.addWidget(QLabel(_("分辨率：")), 1, 0)
        self.cb_resolution = QComboBox()
        # 640x480 is native on virtually every webcam and the safest 30fps
        # mode on USB 2.0. Higher tiers are 16:9 and useful when the camera
        # supports them natively (DSLR + capture card, USB 3.0 webcams).
        self._resolution_options = [
            ("640 x 480", (640, 480)),
            ("960 × 540（准高清）", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 × 1080（全高清）", (1920, 1080)),
        ]
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == cur), 0)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.cb_resolution.setToolTip(
            "摄像头采集分辨率。实际分辨率可能被自动协商到最接近的档位，"
            "真实尺寸会打印在控制台。下次启动实时换脸时生效。\n\n"
            "640x480 几乎所有摄像头都原生支持，是最稳妥的 30 帧选择；"
            "准高清一般也可以；USB 2.0 摄像头跑高清/全高清通常只有约 10 帧。"
        )
        grid.addWidget(self.cb_resolution, 1, 1, 1, 2)

        # Row 2: face detection size
        grid.addWidget(QLabel(_("检测精度：")), 2, 0)
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v} x {v}")
        cur_det = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
        # Normalize to a valid option first so .index() can never raise: fall back
        # to the default size, then to the last (highest) option if even that is
        # somehow missing.
        if cur_det not in self._det_size_options:
            cur_det = (modules.globals.DEFAULT_DET_SIZE
                       if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                       else self._det_size_options[-1])
        idx = self._det_size_options.index(cur_det)
        self.cb_det_size.setCurrentIndex(idx)
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(
            "人脸检测的输入分辨率。数值越小速度越快，但距离稍远的人脸可能识别不到。"
            "下次识别人脸时生效。"
        )
        grid.addWidget(self.cb_det_size, 2, 1, 1, 2)

        grid.setColumnStretch(1, 1)
        return card

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            # Pre-translate the static template; append the dynamic size after
            # so the message still participates in localization (update_status
            # only translates whole-string keys).
            update_status(
                _("采集分辨率已设置（下次启动实时换脸生效）：")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("检测尺寸已设置（下次识别人脸时生效）：")
                + f" {v}x{v}"
            )

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        """底部状态区同一时间只显示一处，避免重复两行。

        规则：
          · 处理中 → 只显示绿色进度条，任务名与进度都写在进度条上
          · 刚处理完成 → 进度条保持满格并显示结果，直到用户做下一步操作
          · 其余情况 → 只显示文字状态行
        """
        self._status_label.setText(text)
        state = getattr(self, "_proc_state", "idle")

        if state == "running":
            # 处理中：隐藏文字状态行，把当前任务写进进度条
            self._status_label.hide()
            bar = getattr(self, "_progress", None)
            if bar is not None:
                bar.setVisible(True)
                stage = text.replace("%", "%%")[:30]
                if bar.maximum() > 0:
                    bar.setFormat(stage + "  %v/%m 帧（%p%）")
                else:
                    bar.setFormat(stage)
        elif state == "done":
            # 完成结果还挂在进度条上；这时收到新消息说明用户已经开始别的操作 → 收起
            self._proc_state = "idle"
            self.hide_progress()
            self._status_label.show()
        else:
            self._status_label.show()

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("请选择源人脸照片"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            # 图片/视频页与实时换脸页两个预览框一起刷新
            self._refresh_source_views()
            self.set_status("已选择源人脸：%s" % os.path.basename(path))
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self._refresh_source_views()
            self.set_status("源人脸已清除，请重新选择")

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        # 按左侧导航当前选中的模式决定过滤器：图片模式只列图片，视频模式只列视频
        if self._current_mode == "video":
            title = "请选择要换脸的目标视频"
            file_filter = _VIDEO_FILE_FILTER
        else:
            title = "请选择要换脸的目标图片"
            file_filter = _IMAGE_FILE_FILTER
        path, _filter = QFileDialog.getOpenFileName(
            self, _(title),
            _RECENT_TARGET_DIR or "",
            file_filter,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
            self.set_status("已选择目标图片：%s" % os.path.basename(path))
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
            self.set_status("已选择目标视频：%s" % os.path.basename(path))
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(self._target_drop_text())
            self.set_status("目标文件已清除，请重新选择")

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/random-person.jpeg",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"应为图片，实际收到 {content_type!r}")
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            staging_path = f"{temp_path}.part"
            # Download into a staging file and validate there. Writing straight
            # to temp_path would destroy the image currently in use whenever a
            # later fetch comes back bad, and content-type alone does not prove
            # the bytes decode.
            try:
                with open(staging_path, "wb") as f:
                    f.write(response.content)
                pixmap = render_image_preview(staging_path, (200, 200))
                if imread_unicode(staging_path) is None:
                    raise ValueError("下载的图片无法解析")
                os.replace(staging_path, temp_path)
            except Exception:
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
                raise
            modules.globals.source_path = temp_path
            # 两个页面的源人脸预览一起刷新（这里已渲染好 pixmap，直接复用）
            for lb in self._source_preview_labels():
                lb.setPixmap(pixmap)
                lb.setText("")
            self.set_status("已获取随机人脸，可直接开始换脸")
        except Exception as exc:
            print(f"获取随机人脸失败：{exc}")
            self.set_status("获取随机人脸失败，请检查网络后重试")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        # 源人脸变了，两个页面的预览框都要同步
        self._refresh_source_views()
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.target_label.setText("")
        self.set_status("已交换源人脸与目标")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        # 选项文案带了括号说明，这里按前缀匹配，避免以后改文案又要同步改 key_map
        def _bucket(text: str) -> str:
            if text.startswith("GFPGAN"):
                return "gfpgan"
            if text.startswith("GPEN-512"):
                return "gpen512"
            if text.startswith("GPEN-256"):
                return "gpen256"
            return "off"

        key_map = {
            "off": None,
            "gfpgan": "face_enhancer",
            "gpen512": "face_enhancer_gpen512",
            "gpen256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(_bucket(choice))
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("融合度为 0%，已关闭换脸效果。")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("融合度已设为 100%。")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"融合度已设为 {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"锐化度已设为 {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("请先完成当前弹窗")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("正在识别人脸...")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("正在识别人脸...")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("目标画面中未检测到人脸")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("保存换脸图片"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("保存换脸视频"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            # 关键：处理放到后台线程跑，主线程保持响应
            # 之前是同步调用 self._start_cb()，整个处理期间 UI 线程被占死，
            # 界面无法重绘 → 按钮文字不变、进度条不动、看着像卡死。
            self._begin_processing()
            self._run_processing_async()

    # ── 后台处理线程 ────────────────────────────────────────────────────

    def _run_processing_async(self) -> None:
        """在后台线程执行换脸，处理完通过信号回到 UI 线程收尾。"""
        start_cb = self._start_cb

        class _Worker(QThread):
            done = Signal(bool, str)   # (是否成功, 错误信息)

            def run(self_inner):
                ok, err = True, ""
                try:
                    start_cb()
                except Exception as exc:      # noqa: BLE001 - 兜底，避免线程静默死亡
                    ok, err = False, str(exc)
                    # 界面上只显示一行错误，完整堆栈写进 runtime.log 便于排查
                    import traceback
                    traceback.print_exc()
                self_inner.done.emit(ok, err)

        self._worker = _Worker(self)
        # 保留引用：QThread 对象被 GC 回收而线程还在跑，Qt 会直接崩溃
        self._workers = getattr(self, "_workers", [])
        self._workers.append(self._worker)
        worker = self._worker

        def _on_done(ok: bool, err: str):
            if not ok:
                # 失败：隐藏进度条，用红色状态文字说明原因（此时没有"进度"可言）
                self._proc_state = "idle"
                self._end_processing()
                self._status_label.setStyleSheet("color:#EF4444;")
                self.set_status("处理失败：%s" % err)
            elif modules.globals.output_path and os.path.isfile(modules.globals.output_path):
                size_mb = os.path.getsize(modules.globals.output_path) / 1024.0 / 1024.0
                name = os.path.basename(modules.globals.output_path)
                # 完成态由进度条承载：满格 + 结果文字，一直保留到用户做下一步操作
                self._proc_state = "done"
                self._end_processing(hide_bar=False)
                self._show_done_on_bar("处理完成：%s（%.2f MB）" % (name, size_mb))
            else:
                self._proc_state = "idle"
                self._end_processing()
                self.set_status("处理结束，但未生成输出文件，请查看日志。")
            # 线程已跑完，安全释放
            worker.wait(3000)
            if worker in self._workers:
                self._workers.remove(worker)

        self._worker.done.connect(_on_done)
        self._worker.start()

    # ── 处理中的 UI 状态 ────────────────────────────────────────────────

    def _begin_processing(self) -> None:
        """处理开始：锁定按钮 + 进度条接管底部状态区。"""
        self._proc_state = "running"
        self._status_label.setStyleSheet("")   # 清掉上次失败留下的红字
        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在换脸，请稍候…")
        self.btn_preview.setEnabled(False)
        # 「打开目录」始终可点：转换前打开目标目录，完成后打开输出目录
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)          # 不确定进度（滚动），拿到帧数后变确定进度
        self._progress.setFormat("正在准备处理，请稍候…")
        # 走 set_status：处理中的文字会同步到进度条，并隐藏文字状态行
        self.set_status("正在准备处理，请稍候…")

    def _end_processing(self, hide_bar: bool = True) -> None:
        """处理结束（成功或异常都执行）：恢复按钮 + 隐藏进度条。

        hide_bar=False 时保留进度条（成功时由 _show_done_on_bar 展示结果）。
        """
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始换脸")
        self.btn_preview.setEnabled(True)
        if hide_bar:
            self.hide_progress()

    def _show_done_on_bar(self, text: str) -> None:
        """把完成结果画在进度条上（满格 + 结果文字），并隐藏文字状态行。

        完成后不自动隐藏：用户能一直看到「处理完成 + 文件名 + 大小」，
        直到选新的文件或开始下一次处理时才收起。
        """
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        # Qt 格式串里 % 是占位符，文件名里若含 % 必须转义
        self._progress.setFormat(text.replace("%", "%%"))
        self._status_label.setText(text)
        self._status_label.hide()

    def closeEvent(self, event) -> None:
        """关闭窗口的保护：处理线程还在跑时不允许直接退出。

        为什么必须拦：QThread 正在执行时被销毁，Qt 会直接终止进程（崩溃）。
        原来处理是跑在 UI 线程上的，界面卡死反而不会触发这个场景；
        改成后台线程后窗口可响应了，就必须在这里挡住。
        """
        running = any(w.isRunning() for w in getattr(self, "_workers", []))
        if running:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "正在处理",
                "换脸任务还没有完成，请等待处理结束后再关闭窗口。",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()
        else:
            update_status("请先选择源人脸和目标文件")
        # 开/关之后同步按钮形态
        self._sync_preview_btn()

    def _sync_preview_btn(self) -> None:
        """预览按钮双态：窗口开着显示「关闭预览」（红底），关着显示「预览」。

        说明：objectName 变了必须 unpolish/polish 才会重新套用 QSS，
        否则颜色不会跟着变。
        """
        btn = getattr(self, "btn_preview", None)
        if btn is None:
            return
        visible = _PREVIEW is not None and _PREVIEW.isVisible()
        btn.setText(_("关闭预览") if visible else _("预览"))
        btn.setObjectName("danger" if visible else "secondary")
        btn.setToolTip(
            _("关闭预览窗口") if visible else _("显示/隐藏处理结果的预览窗口")
        )
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _on_open_dir(self) -> None:
        """打开输出目录。

        优先级：本次保存位置 → 输出路径 → 目标文件所在目录。
        这样转换前（还没保存过）也能打开，用户可以先看看素材在哪。
        """
        global _RECENT_OUTPUT_DIR
        path = _RECENT_OUTPUT_DIR
        if not path and modules.globals.output_path:
            path = os.path.dirname(modules.globals.output_path)
        if not path and modules.globals.target_path:
            path = os.path.dirname(modules.globals.target_path)
        if not path or not os.path.isdir(path):
            update_status("还没有可打开的目录：请先选择目标文件或设置保存位置")
            return
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            update_status("已打开目录：%s" % path)
        except Exception as exc:
            update_status("打开目录失败：%s" % exc)

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("没有可用摄像头")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("人脸对应窗口已打开")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("请先选择源人脸照片")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("预览"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("正在处理...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_frame = imread_unicode(modules.globals.source_path)
        if source_frame is None:
            update_status(
                f"无法读取源人脸照片：{modules.globals.source_path}"
            )
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(get_one_face(source_frame), temp_frame)
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("处理完成！")

    def closeEvent(self, event) -> None:
        """用户点预览窗口右上角关闭时，把底部按钮恢复成「预览」。

        不加这段的话按钮会一直停在"关闭预览"，用户会以为预览还开着。
        """
        try:
            if _MAIN is not None:
                _MAIN._sync_preview_btn()
        except Exception:
            pass
        super().closeEvent(event)


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        reported_source_error = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        det_interval = max(1, round(self._fps * 0.08))

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    source_frame = imread_unicode(modules.globals.source_path)
                    if source_frame is None:
                        # Clear the cached path along with the image. It keeps
                        # a source that is only transiently unreadable (still
                        # being written, on a volume that just went away) from
                        # being given up on, and it keeps a bad pick from
                        # sticking: with the path cached, A -> unreadable B -> A
                        # never reloaded A, so swapping stayed off. Retrying
                        # every frame is cheap; the message is reported once per
                        # path so it cannot flood the status line.
                        source_image = None
                        last_source_path = None
                        if reported_source_error != modules.globals.source_path:
                            reported_source_error = modules.globals.source_path
                            update_status(
                                f"无法读取源人脸照片：{modules.globals.source_path}"
                            )
                    else:
                        last_source_path = modules.globals.source_path
                        reported_source_error = None
                        source_image = get_one_face(source_frame)

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask needs it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"帧率 {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("实时预览")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        if not self._cap.start(req_w, req_h, 60):
            update_status("摄像头启动失败")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        # __init__ can bail out before these exist (e.g. the camera fails to open),
        # and closeEvent still runs.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        # Shutdown has to actually complete. Qt aborts the process when a
        # running QThread is destroyed, and holding a straggler in a module
        # global only moves that abort to interpreter exit. Both loops re-check
        # the stop flag between bounded operations (a camera read, a 50ms queue
        # get, one inference), so waiting terminates; a first-time model load
        # is simply slow, which is what the grace period absorbs quietly.
        for name in ("_capture_worker", "_processing_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            if not worker.wait(WORKER_SHUTDOWN_GRACE_MS):
                print(
                    f"[webcam] still waiting for {name[1:]} to stop...",
                    flush=True,
                )
                worker.wait()
        # Only now: releasing the capture while the capture thread could still
        # be inside cap.read() is not safe.
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("人脸对应设置"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("确认"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("选择源图"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("请选择源人脸照片"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("上次上传的图片中未检测到人脸")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("至少需要一组「源人脸 + 目标人脸」对应")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("人脸对应设置"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("添加"), self._on_add),
            (_("清除"), self._on_clear),
            (_("确认"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("选择源图"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("选择目标图"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("请选择源人脸照片"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("上次上传的图片中未检测到人脸")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("请先设置人脸对应关系")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("已清除全部对应关系")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("对应关系设置成功")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("至少需要一组「源人脸 + 目标人脸」对应")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _close_live_windows() -> None:
    """Close every window that owns worker threads, so the threads stop."""
    for win in (_WEBCAM_PREVIEW, _LIVE_MAPPER):
        if win is None:
            continue
        try:
            win.close()
        except Exception:
            pass


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def _find_app_icon() -> Optional[str]:
    """返回可用的图标文件路径（找不到返回 None）。

    包内安装后图标在 assets/app.ico；开发目录里可能只有 tools/launcher/app.ico。
    """
    import modules.paths as _paths

    for parts in (("assets", "app.ico"), ("tools", "launcher", "app.ico")):
        p = os.path.join(_paths.ROOT_DIR, *parts)
        if os.path.isfile(p):
            return p
    return None


def _setup_app_icon() -> None:
    """设置窗口图标 + 任务栏图标。

    为什么要分两步：
      1. setWindowIcon 只影响窗口左上角与 Alt+Tab 缩略图；
      2. Windows 任务栏是按"进程"分组的。我们的进程是 pythonw.exe，
         所以任务栏会显示 Python 的图标而不是 TSAIFACE 的。
         调用 SetCurrentProcessExplicitAppUserModelID 给进程一个独立身份，
         Windows 才会用我们自己的图标（run.py 里也提前设了一次）。
    """
    from PySide6.QtGui import QIcon

    icon_path = _find_app_icon()
    if icon_path is None:
        return

    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(QIcon(icon_path))

    # Windows 任务栏：给进程独立 AppUserModelID（必须与 run.py 保持一致）
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "TSAIFACE.Desktop.1"
            )
        except Exception:
            pass


def _install_fonts() -> None:
    """注册界面字体。

    为什么要显式注册：
      1. 整合包要发给各种 Windows 机器，个别精简版系统缺「微软雅黑」；
      2. 打包成 exe 后工作目录可能变化，Qt 有时找不到系统字体目录，
         表现为界面上的中文全部显示成方块（□□□）。
    做法：优先用包内自带的字体文件（assets/fonts），没有再退回系统字体。
    这样即使对方电脑没有中文字体，也不会出现方块。
    """
    from PySide6.QtGui import QFontDatabase, QFont
    import modules.paths as _paths

    # 包内自带字体目录（打包时把 msyh.ttc 放进去，规避系统缺字体）
    bundled = os.path.join(_paths.ROOT_DIR, "assets", "fonts")
    if os.path.isdir(bundled):
        for fn in sorted(os.listdir(bundled)):
            if fn.lower().endswith((".ttf", ".ttc", ".otf")):
                QFontDatabase.addApplicationFont(os.path.join(bundled, fn))

    # 兜底：显式指定一个确定存在的中文字体族
    # 注意 QFont.setFamily 是实例方法，必须先建 QFont 对象再设置，
    # 并通过 QApplication.setFont 生效（否则中文仍可能 fallback 成方块）
    families = QFontDatabase.families()
    for name in ("Microsoft YaHei", "微软雅黑", "SimHei", "黑体", "SimSun", "宋体"):
        if name in families:
            app = QApplication.instance()
            if app is not None:
                f = app.font()
                f.setFamily(name)
                app.setFont(f)
            break


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE, _THEME

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _install_fonts()
    _setup_app_icon()
    # 主题：读取上次的选择（默认深色），再应用到全局样式
    _THEME = load_theme()
    apply_theme(_THEME)

    _BRIDGE = _UIBridge()
    def _destroy_with_cleanup(*args, **kwargs):
        # core.destroy() ends in a bare quit(), which raises SystemExit and tears
        # the interpreter down without unwinding Qt. Destroying a running QThread
        # is fatal in Qt, so stop the live-preview workers before that happens.
        _close_live_windows()
        return destroy(*args, **kwargs)

    _MAIN = MainWindow(start, _destroy_with_cleanup)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)
    # 处理进度：core 每 N 帧上报一次，刷新底部进度条
    _BRIDGE.progressChanged.connect(_MAIN.set_progress)

    return _Window(_APP, _MAIN)
