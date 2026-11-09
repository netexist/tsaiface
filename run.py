#!/usr/bin/env python3

import os
import sys

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")


def _ensure_std_streams() -> None:
    """给「无控制台」启动方式补上标准输出/错误流。

    为什么必须做：
      整合包用 TSAIFACE.exe → pythonw.exe 启动，pythonw 没有控制台，
      sys.stdout / sys.stderr 都是 None。任何 print(...) 或第三方进度条
      （tqdm 默认写 sys.stderr）都会抛出
      AttributeError: 'NoneType' object has no attribute 'write'，
      视频处理会因此直接中断。
    做法：把输出接到项目目录下的 runtime.log（每次启动覆盖，只保留本次运行），
          目录不可写时退回系统黑洞，保证程序永远能跑起来。
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    stream = None
    try:
        stream = open(
            os.path.join(project_root, "runtime.log"),
            "w", encoding="utf-8", errors="replace", buffering=1,
        )
    except Exception:
        stream = None
    if stream is None:
        try:
            stream = open(os.devnull, "w", encoding="utf-8")
        except Exception:
            return          # 实在无法创建，保持原状（比崩溃好）
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


_ensure_std_streams()

# 单实例：同一个程序只允许运行一个。必须在这里（创建任何窗口之前）判断。
_MUTEX_HANDLE = None


def _single_instance_guard() -> bool:
    """只允许运行一个 TSAIFACE 实例，返回 True 表示可以继续启动。

    为什么锁必须在 Python 侧持有：
      启动器 TSAIFACE.exe 只是"拉起 pythonw 后观察 6 秒就退出"的引导程序，
      它持有的互斥锁会随它退出而释放，之后再双击照样能开出第二个窗口。
      只有让真正长期驻留的进程（pythonw）持有锁，才能真正挡住多开。
    """
    global _MUTEX_HANDLE
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        # use_last_error=True 才能用 ctypes.get_last_error 准确读到 ERROR_ALREADY_EXISTS
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # 不加 Global\ 前缀：普通用户创建 Global 命名对象可能被拒绝访问，
        # 那样句柄为 NULL 就只能放行，等于单实例没生效。
        # 默认的会话级命名（Local）足够挡住"同一桌面重复双击"，且不需要任何权限。
        handle = kernel32.CreateMutexW(None, False, "TSAIFACE_SingleInstance")
        if not handle:
            return True          # 连锁句柄都拿不到就放行，不要因为锁把程序挡死
        if ctypes.get_last_error() == 183:      # ERROR_ALREADY_EXISTS
            try:
                ctypes.WinDLL("user32").MessageBoxW(
                    0,
                    "TSAIFACE 已经在运行中，请查看任务栏。\n\n"
                    "如果找不到窗口，请在任务管理器中结束 pythonw.exe 后重试。",
                    "已经在运行",
                    0x40,             # MB_ICONINFORMATION
                )
            except Exception:
                pass
            return False
        _MUTEX_HANDLE = handle   # 持有到进程结束，系统自动释放
        return True
    except Exception:
        return True


if not _single_instance_guard():
    sys.exit(0)

# Windows 任务栏图标：先给进程一个独立的应用身份（AppUserModelID）。
# 为什么必须在这里做：Windows 任务栏按进程分组，我们的进程是 pythonw.exe，
# 不设置的话任务栏会用 Python 的图标。这个 API 必须在创建任何窗口之前调用，
# 所以放在导入 PySide6 / 创建 QApplication 之前。
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TSAIFACE.Desktop.1"
        )
    except Exception:
        pass

# On Windows, register NVIDIA CUDA DLL directories so onnxruntime-gpu can
# find cuDNN/cublas. Python 3.8+ ignores PATH for extension-module native deps —
# os.add_dll_directory() is required. Also keep PATH for child processes/ffmpeg.
if sys.platform == "win32":
    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _candidate_dirs = []
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            _candidate_dirs.append(_torch_lib)
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    _candidate_dirs.append(_bin_dir)
        for _d in _candidate_dirs:
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

    # On Windows, register OpenVINO DLL directories so onnxruntime's
    # OpenVINOExecutionProvider can find openvino.dll.  This must happen
    # before any ONNX InferenceSession is created.  Failure is non-fatal:
    # OpenVINO simply isn't installed, and onnxruntime will fall back to CPU.
    try:
        from onnxruntime.tools.add_openvino_win_libs import (  # type: ignore[import-untyped]  # noqa: E501
            add_openvino_libs_to_path,
        )
        add_openvino_libs_to_path()
    except ImportError:
        # onnxruntime build without the OpenVINO tooling module — no-op.
        pass
    except FileNotFoundError:
        # OpenVINO site-packages dir absent — no-op.
        pass
    except SystemExit as exc:
        # add_openvino_libs_to_path() calls sys.exit() when OpenVINO libs
        # can't be located (e.g. OPENVINO_LIB_PATHS unset).  Log the message
        # it raised with so the failure is visible, but keep startup alive.
        print(
            f"[startup] OpenVINO DLL registration skipped: {exc}",
            flush=True,
        )

# On Linux, pre-load NVIDIA shared libraries (cuDNN, cuBLAS, nvrtc...) shipped
# inside the venv via pip wheels (nvidia-cudnn-cu12, etc.). LD_LIBRARY_PATH
# cannot be set after Python starts, so we use ctypes.CDLL with RTLD_GLOBAL
# instead. This makes symbols available to onnxruntime when it dlopens its
# CUDA provider.
if sys.platform.startswith("linux"):
    import ctypes
    import glob
    _py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _site_packages_candidates = [
        os.path.join(project_root, "venv", "lib", _py_lib, "site-packages"),
        os.path.join(sys.prefix, "lib", _py_lib, "site-packages"),
    ]
    for _sp in _site_packages_candidates:
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if not os.path.isdir(_nvidia_dir):
            continue
        for _pkg in os.listdir(_nvidia_dir):
            _lib_dir = os.path.join(_nvidia_dir, _pkg, "lib")
            if not os.path.isdir(_lib_dir):
                continue
            # Also expose the directory to child processes, without
            # duplicating an entry that is already present.
            _ldp = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_dir not in _ldp.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = (
                    _lib_dir + (os.pathsep + _ldp if _ldp else "")
                )
            for _so in sorted(glob.glob(os.path.join(_lib_dir, "lib*.so*"))):
                try:
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        break

from modules import platform_info
platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    core.run()
