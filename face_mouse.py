import sys
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import pyperclip

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import sounddevice as sd
import queue
import whisper

from PyQt5 import QtCore, QtGui, QtWidgets


# =========================
#   阻尼滤波器
# =========================

@dataclass
class FilterState:
    x: float = 0.0
    y: float = 0.0


class DampedFilter:
    """
    一阶低通 / 阻尼滤波：
    new = alpha * prev + (1 - alpha) * measurement
    alpha 越大越平滑（抖动少，但反应慢）
    """
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self.state = FilterState()

    def reset(self, x: float, y: float):
        self.state = FilterState(x, y)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        self.state.x = self.alpha * self.state.x + (1 - self.alpha) * x
        self.state.y = self.alpha * self.state.y + (1 - self.alpha) * y
        return self.state.x, self.state.y


# =========================
#   面部鼠标控制器
# =========================

class FaceMouseController:
    def __init__(self, camera_index: int = 0, alpha: float = 0.8):
        # 摄像头
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 屏幕信息
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.filter = DampedFilter(alpha=alpha)
        # 鼠标移动放大系数（越大，小幅头动 → 大幅鼠标移动）
        # 鼠标移动放大系数（如果已经加过可以忽略）
        self.gain = 3.0   # 可按自己感觉调：2.0 ~ 4.0

        # 头部“居中姿势”的鼻子位置（像素）
        self.center_x: Optional[float] = None
        self.center_y: Optional[float] = None

        # 最近一帧的鼻子位置（用于“居中”时记录）
        self.last_nose = None

        # 嘴巴 & 点击控制
        self.mouth_open = False
        self.mouth_open_start: Optional[float] = None
        self.mouth_right_clicked_this_open = False
        self.last_left_click_time = 0.0
        self.left_click_min_interval = 0.3  # 左键最小间隔
        self.long_open_threshold = 2.0      # 长时间张嘴阈值（秒）

    def _get_landmarks(self, frame_rgb) -> Optional[np.ndarray]:
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame_rgb.shape
        pts = []
        for lm in face_landmarks.landmark:
            pts.append((lm.x * w, lm.y * h, lm.z))
        return np.array(pts, dtype=np.float32)

    def _estimate_head_roll(self, pts: np.ndarray) -> float:
        # 左右耳附近点估算头部横滚角
        left_ear_idx = 234
        right_ear_idx = 454
        if max(left_ear_idx, right_ear_idx) >= len(pts):
            return 0.0
        p_left = pts[left_ear_idx]
        p_right = pts[right_ear_idx]
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)

    def _nose_position(self, pts: np.ndarray) -> Tuple[float, float]:
        # 鼻尖
        nose_idx = 1
        if nose_idx >= len(pts):
            nose_idx = 4
        nose = pts[nose_idx]
        return float(nose[0]), float(nose[1])

    def _mouth_open_ratio(self, pts: np.ndarray) -> float:
        # 嘴巴张开程度 = 上下唇距离 / (鼻子到下巴距离)
        upper_idx = 13   # 上唇
        lower_idx = 14   # 下唇
        chin_idx = 152   # 下巴
        nose_idx = 1     # 鼻子

        if max(upper_idx, lower_idx, chin_idx, nose_idx) >= len(pts):
            return 0.0

        upper = pts[upper_idx]
        lower = pts[lower_idx]
        chin = pts[chin_idx]
        nose = pts[nose_idx]

        mouth_dist = float(np.linalg.norm(upper[:2] - lower[:2]))
        ref_dist = float(np.linalg.norm(nose[:2] - chin[:2]))
        if ref_dist < 1e-3:
            return 0.0
        return mouth_dist / ref_dist

    def _map_to_screen(self, x: float, y: float, frame_shape) -> Tuple[int, int]:
        """
        将鼻尖坐标相对“居中姿势”放大后，映射到屏幕坐标。

        - 如果还没设置 center_x/center_y，就默认用画面中心作为居中姿势。
        - 设置了居中姿势后，这个姿势对应屏幕中心。
        """
        h, w, _ = frame_shape

        # 如果用户还没“居中”，默认用画面中心
        cx = self.center_x if self.center_x is not None else w / 2.0
        cy = self.center_y if self.center_y is not None else h / 2.0

        # 相对中心的偏移，归一化到 [-1, 1]
        dx_norm = (x - cx) / (w / 2.0)  # 左右偏移
        dy_norm = (y - cy) / (h / 2.0)  # 上下偏移

        # 放大系数：决定“头动一点 → 鼠标动多少”
        dx_norm *= self.gain
        dy_norm *= self.gain

        # 限制到 [-1, 1] 防止飞到屏幕外
        dx_norm = max(-1.0, min(1.0, dx_norm))
        dy_norm = max(-1.0, min(1.0, dy_norm))

        # 映射到 [0, 1]，0.5 表示屏幕中心
        nx = 0.5 + dx_norm / 2.0
        ny = 0.5 + dy_norm / 2.0

        # 最后变成屏幕像素坐标
        sx = int(self.screen_w * nx)
        sy = int(self.screen_h * ny)
        return sx, sy


    def set_center_from_last(self):
        """
        把最近一帧的鼻子位置设为“居中姿势”。
        需要在有脸、检测到鼻子以后调用。
        """
        if self.last_nose is None:
            print("[FaceMouse] No nose position yet, cannot set center.")
            return

        nose_x, nose_y, _ = self.last_nose
        self.center_x = nose_x
        self.center_y = nose_y
        print(f"[FaceMouse] Center set to ({self.center_x:.1f}, {self.center_y:.1f})")



    def update(self, frame, enabled: bool = True, control_status: str = "ON") -> Tuple[np.ndarray, Optional[Tuple[int, int]], float]:
        """
        enabled == False 时，只画图但不移动鼠标、不点击
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = self._get_landmarks(frame_rgb)
        head_roll = 0.0
        screen_pos = None

        if pts is not None:
            # 鼻子 -> 屏幕坐标
            nose_x, nose_y = self._nose_position(pts)
            self.last_nose = (nose_x, nose_y, frame.shape)
            head_roll = self._estimate_head_roll(pts)
            raw_sx, raw_sy = self._map_to_screen(nose_x, nose_y, frame.shape)

            if self.filter.state.x == 0.0 and self.filter.state.y == 0.0:
                self.filter.reset(raw_sx, raw_sy)
            sx, sy = self.filter.update(raw_sx, raw_sy)
            screen_pos = (int(sx), int(sy))

            # 面部控制启用才移动鼠标
            if enabled:
                pyautogui.moveTo(screen_pos[0], screen_pos[1])

            # 嘴巴张开检测：短开=左键，长开=右键
            ratio = self._mouth_open_ratio(pts)
            now = time.time()
            open_threshold = 0.07
            close_threshold = 0.05

            if ratio > open_threshold:
                # 嘴巴处于张开状态
                if not self.mouth_open:
                    # 新一次张嘴开始
                    self.mouth_open = True
                    self.mouth_open_start = now
                    self.mouth_right_clicked_this_open = False
                else:
                    # 已经张开了一段时间，检查是否需要右键
                    if (self.mouth_open_start is not None and
                            not self.mouth_right_clicked_this_open and
                            now - self.mouth_open_start >= self.long_open_threshold):
                        if enabled:
                            pyautogui.click(button="right")
                        self.mouth_right_clicked_this_open = True
                        self.last_left_click_time = now  # 避免紧接着再触发左键
            else:
                # 嘴巴闭合，如果之前是张开的，判断这次张嘴持续时间
                if self.mouth_open:
                    duration = 0.0
                    if self.mouth_open_start is not None:
                        duration = now - self.mouth_open_start
                    # 短张嘴且没有触发右键 -> 左键单击
                    if (not self.mouth_right_clicked_this_open and
                            0.1 <= duration < self.long_open_threshold and
                            enabled and
                            now - self.last_left_click_time >= self.left_click_min_interval):
                        pyautogui.click(button="left")
                        self.last_left_click_time = now
                # 重置状态
                self.mouth_open = False
                self.mouth_open_start = None
                self.mouth_right_clicked_this_open = False

            # 在画面上画关键点
            for idx in [1, 13, 14, 234, 454]:
                if idx < len(pts):
                    cx = int(pts[idx][0])
                    cy = int(pts[idx][1])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # 显示信息
            cv2.putText(frame, f"Head roll: {head_roll:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth open: {self.mouth_open}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Control: {control_status}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, screen_pos, head_roll

    def release(self):
        self.cap.release()
        self.face_mesh.close()


# =========================
#   Whisper 线程
# =========================

class WhisperWorker(QtCore.QThread):
    text_recognized = QtCore.pyqtSignal(str)   # 识别出的文本
    command_detected = QtCore.pyqtSignal(str)  # "pause" / "resume"

    def __init__(self, model_name: str = "small", language_code: str = "zh", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.language_code = language_code
        self.model = None
        self.queue = queue.Queue()
        self.running = True

    def set_language(self, language_code: str):
        self.language_code = language_code

    def run(self):
        print(f"[Whisper] Loading model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)

        while self.running:
            try:
                audio_block = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if audio_block is None or not self.running:
                continue

            # 转单声道 float32
            if audio_block.ndim > 1:
                audio = audio_block.mean(axis=1)
            else:
                audio = audio_block
            audio = audio.astype(np.float32)

            try:
                result = self.model.transcribe(
                    audio,
                    fp16=False,
                    language=self.language_code
                )
            except Exception as e:
                print("Whisper error:", e)
                continue

            text = result.get("text", "").strip()
            if not text:
                continue

            # 发信号给 UI
            self.text_recognized.emit(text)

            # 语音指令识别
            if self.language_code == "zh":
                if "暂停" in text or "停止" in text:
                    self.command_detected.emit("pause")
                if "继续" in text or "开始" in text:
                    self.command_detected.emit("resume")
            else:
                lower = text.lower()
                if "pause" in lower or "stop" in lower:
                    self.command_detected.emit("pause")
                if "resume" in lower or "start" in lower:
                    self.command_detected.emit("resume")

            # 在当前焦点窗口输入文字
            try:
                pyperclip.copy(text + " ")
                pyautogui.hotkey("ctrl", "v")
            except Exception as e:
                print("Paste error:", e)

    def add_audio(self, audio: np.ndarray):
        if self.running:
            self.queue.put(audio)

    def stop(self):
        self.running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


# =========================
#   麦克风录音（带能量阈值）
# =========================

class AudioRecorder(QtCore.QObject):
    level_state_changed = QtCore.pyqtSignal(str)  # "idle" / "listening"

    def __init__(self, worker: WhisperWorker, samplerate: int = 16000,
                 block_duration: float = 5.0, level_threshold: float = 0.02, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.samplerate = samplerate
        self.block_size = int(samplerate * block_duration)
        self.stream = None
        self.level_threshold = level_threshold  # RMS 小于该值时视为静音

    def set_threshold(self, threshold: float):
        self.level_threshold = threshold

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        # 计算音量 RMS，做一个简单的“去噪 + 静音检测”
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        if rms < self.level_threshold:
            # 静音：不送去 STT
            self.level_state_changed.emit("idle")
            return
        else:
            self.level_state_changed.emit("listening")
            self.worker.add_audio(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
            blocksize=self.block_size,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# =========================
#   主窗口 UI
# =========================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 默认语言和噪声阈值
        self.language_code = "zh"
        self.noise_threshold = 0.02  # 语音能量阈值

        self.controller = FaceMouseController(alpha=0.8)
        self.face_control_enabled = True  # 是否启用面部鼠标控制

        self.whisper_worker: Optional[WhisperWorker] = None
        self.audio_recorder: Optional[AudioRecorder] = None

        self.init_ui()
        self.init_whisper_audio()
        self.predownload_medium_model()

        # 定时器：刷新摄像头画面
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33fps

        self.on_stt_level_state("idle")

    # ---------- 多语言文本 ----------
    def get_translations(self):
        return {
            "zh": {
                "window_title": "脸控鼠标 + Whisper 语音输入",
                "enable_face_control": "启用面部鼠标控制（Ctrl 或 语音“暂停 / 继续”）",
                "whisper_model": "Whisper 模型：",
                "change_model": "切换模型",
                "sensitivity_label": "灵敏度（滤波 alpha，0.01~0.99，越小越灵敏）：",
                "on_top": "窗口置顶（始终显示在最前）",
                "stt_text_label": "语音识别文本：",
                "clear_text": "清空文本",
                "status_model": "当前 Whisper 模型：{}",
                "language_label": "界面 / 识别语言：",
                "language_zh": "中文",
                "language_en": "English",
                "stt_status_idle": "等待说话",
                "stt_status_listening": "正在语音识别...",
                "noise_label": "语音能量阈值（0.001~0.1，越大越不易触发识别）：",
                "center_button": "设置当前头部姿势为屏幕中心",
            },
            "en": {
                "window_title": "Face Mouse + Whisper Speech Input",
                "enable_face_control": "Enable face mouse control (Ctrl or voice 'pause / resume')",
                "whisper_model": "Whisper model:",
                "change_model": "Switch model",
                "sensitivity_label": "Sensitivity (filter alpha, 0.01~0.99, smaller = more sensitive):",
                "on_top": "Always on top (keep this window in front)",
                "stt_text_label": "Speech recognition text:",
                "clear_text": "Clear text",
                "status_model": "Current Whisper model: {}",
                "language_label": "UI / STT language:",
                "language_zh": "Chinese",
                "language_en": "English",
                "stt_status_idle": "Waiting for speech",
                "stt_status_listening": "Recognizing speech...",
                "noise_label": "Audio energy threshold (0.001~0.1, larger = harder to trigger STT):",
                "center_button": "Set current head pose as screen center",
            },
        }

    def retranslate_ui(self):
        t = self.get_translations()[self.language_code]
        self.setWindowTitle(t["window_title"])
        self.face_checkbox.setText(t["enable_face_control"])
        self.center_btn.setText(t["center_button"])
        self.model_label.setText(t["whisper_model"])
        self.model_button.setText(t["change_model"])
        self.sens_label.setText(t["sensitivity_label"])
        self.on_top_checkbox.setText(t["on_top"])
        self.stt_text_label.setText(t["stt_text_label"])
        self.clear_btn.setText(t["clear_text"])
        self.language_label.setText(t["language_label"])
        # 更新语言下拉框显示文本
        self.language_combo.blockSignals(True)
        self.language_combo.setItemText(0, t["language_zh"])
        self.language_combo.setItemText(1, t["language_en"])
        self.language_combo.blockSignals(False)
        self.noise_label.setText(t["noise_label"])

        # 状态栏
        if self.whisper_worker is not None:
            self.statusBar().showMessage(t["status_model"].format(self.whisper_worker.model_name))

        # 刷新 STT 状态文字
        self.on_stt_level_state(getattr(self, "_last_stt_state", "idle"))

    # ---------- UI 布局 ----------
    def init_ui(self):
        self.setWindowTitle("Face Mouse + Whisper Speech Input")
        self.resize(1100, 650)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        # 左侧：视频显示
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")

        # 右侧：控制面板
        right_layout = QtWidgets.QVBoxLayout()

        # 面部控制启用
        self.face_checkbox = QtWidgets.QCheckBox()
        self.face_checkbox.setChecked(True)
        self.face_checkbox.stateChanged.connect(self.on_face_checkbox)
        right_layout.addWidget(self.face_checkbox)

        # 居中按钮：把当前头部姿势设为屏幕中心
        self.center_btn = QtWidgets.QPushButton()
        self.center_btn.clicked.connect(self.on_center_clicked)
        right_layout.addWidget(self.center_btn)

        # 语言选择
        lang_layout = QtWidgets.QHBoxLayout()
        self.language_label = QtWidgets.QLabel()
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["中文", "English"])
        self.language_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.language_label)
        lang_layout.addWidget(self.language_combo)
        right_layout.addLayout(lang_layout)

        # Whisper 模型选择
        model_layout = QtWidgets.QHBoxLayout()
        self.model_label = QtWidgets.QLabel()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        self.model_combo.setCurrentText("small")
        self.model_button = QtWidgets.QPushButton()
        self.model_button.clicked.connect(self.change_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_button)
        right_layout.addLayout(model_layout)

        # 灵敏度调节
        self.sens_label = QtWidgets.QLabel()
        right_layout.addWidget(self.sens_label)

        sens_layout = QtWidgets.QHBoxLayout()
        self.sensitivity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)   # 0.001
        self.sensitivity_slider.setMaximum(999)  # 0.999
        self.sensitivity_slider.setValue(800)    # 默认 0.800
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_slider)
        self.sensitivity_edit = QtWidgets.QLineEdit("0.80")
        self.sensitivity_edit.setFixedWidth(60)
        self.sensitivity_edit.editingFinished.connect(self.on_sensitivity_edit)
        sens_layout.addWidget(self.sensitivity_slider)
        sens_layout.addWidget(self.sensitivity_edit)
        right_layout.addLayout(sens_layout)

        # 语音能量阈值
        self.noise_label = QtWidgets.QLabel()
        right_layout.addWidget(self.noise_label)

        noise_layout = QtWidgets.QHBoxLayout()
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setMinimum(1)    # 0.001
        self.noise_slider.setMaximum(100)  # 0.100
        self.noise_slider.setValue(int(self.noise_threshold * 1000))
        self.noise_slider.valueChanged.connect(self.on_noise_slider)
        self.noise_edit = QtWidgets.QLineEdit(f"{self.noise_threshold:.3f}")
        self.noise_edit.setFixedWidth(60)
        self.noise_edit.editingFinished.connect(self.on_noise_edit)
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_edit)
        right_layout.addLayout(noise_layout)

        # 窗口置顶
        self.on_top_checkbox = QtWidgets.QCheckBox()
        self.on_top_checkbox.stateChanged.connect(self.on_on_top_changed)
        right_layout.addWidget(self.on_top_checkbox)

        # STT 状态
        self.stt_status_label = QtWidgets.QLabel()
        right_layout.addWidget(self.stt_status_label)

        # 语音识别文本框
        self.stt_text_label = QtWidgets.QLabel()
        right_layout.addWidget(self.stt_text_label)
        self.text_edit = QtWidgets.QTextEdit()
        right_layout.addWidget(self.text_edit)

        self.clear_btn = QtWidgets.QPushButton()
        self.clear_btn.clicked.connect(self.text_edit.clear)
        right_layout.addWidget(self.clear_btn)

        right_layout.addStretch()

        # 主布局
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(right_layout)

        central.setLayout(main_layout)

        # 状态栏
        self.statusBar().showMessage("")

        # 应用当前语言文本
        self.retranslate_ui()

    # ---------- Whisper + 音频 ----------
    def init_whisper_audio(self):
        model_name = self.model_combo.currentText()
        self.whisper_worker = WhisperWorker(model_name=model_name, language_code=self.language_code)
        self.whisper_worker.text_recognized.connect(self.on_text_recognized)
        self.whisper_worker.command_detected.connect(self.on_command_detected)
        self.whisper_worker.start()

        self.audio_recorder = AudioRecorder(
            self.whisper_worker,
            samplerate=16000,
            block_duration=5.0,
            level_threshold=self.noise_threshold,
            parent=self,
        )
        self.audio_recorder.level_state_changed.connect(self.on_stt_level_state)
        self.audio_recorder.start()

        t = self.get_translations()[self.language_code]
        self.statusBar().showMessage(t["status_model"].format(model_name))

    def predownload_medium_model(self):
        # 后台预下载 medium 模型，避免切换时等待太久
        import threading

        def worker():
            try:
                print("[Whisper] Pre-downloading 'medium' model in background...")
                m = whisper.load_model("medium")
                del m
                print("[Whisper] 'medium' model is ready.")
            except Exception as e:
                print("[Whisper] Pre-download medium failed:", e)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- 摄像头更新 ----------
    def update_frame(self):
        ret, frame = self.controller.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        control_status = "ON" if self.face_control_enabled else "PAUSED"
        frame_out, _, _ = self.controller.update(
            frame,
            enabled=self.face_control_enabled,
            control_status=control_status
        )

        # OpenCV 图像 -> Qt 显示
        rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    # ---------- 面部控制开关 ----------
    def on_face_checkbox(self, state):
        self.face_control_enabled = (state == QtCore.Qt.Checked)

    def on_center_clicked(self):
        """
        用户点击“居中”按钮：
        把当前检测到的鼻子位置设为面部控制的零点（屏幕中心）。
        """
        if self.controller.last_nose is None:
            # 还没检测到脸的时候点也没用，就简单打印一下
            print("[UI] No nose position yet, cannot set center.")
            return

        self.controller.set_center_from_last()

    def toggle_face_control(self):
        self.face_control_enabled = not self.face_control_enabled
        self.face_checkbox.setChecked(self.face_control_enabled)

    # 捕获 Ctrl 键（窗口激活时）
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Control:
            self.toggle_face_control()
        else:
            super().keyPressEvent(event)

    # ---------- Whisper 文本 & 语音指令 ----------
    @QtCore.pyqtSlot(str)
    def on_text_recognized(self, text: str):
        # 将新识别的文本追加到文本框
        self.text_edit.append(text)

    @QtCore.pyqtSlot(str)
    def on_command_detected(self, cmd: str):
        if cmd == "pause":
            if self.face_control_enabled:
                self.toggle_face_control()
        elif cmd == "resume":
            if not self.face_control_enabled:
                self.toggle_face_control()

    # ---------- STT 状态 ----------
    @QtCore.pyqtSlot(str)
    def on_stt_level_state(self, state: str):
        self._last_stt_state = state
        t = self.get_translations()[self.language_code]
        if state == "listening":
            self.stt_status_label.setText(t["stt_status_listening"])
        else:
            self.stt_status_label.setText(t["stt_status_idle"])

    # ---------- 模型切换 ----------
    def change_model(self):
        new_model = self.model_combo.currentText()
        t = self.get_translations()[self.language_code]
        self.statusBar().showMessage(t["status_model"].format(new_model))

        # 停止旧的录音和线程
        if self.audio_recorder is not None:
            self.audio_recorder.stop()
            self.audio_recorder = None

        if self.whisper_worker is not None:
            self.whisper_worker.stop()
            self.whisper_worker.wait()
            self.whisper_worker = None

        # 创建新的 worker
        self.whisper_worker = WhisperWorker(model_name=new_model, language_code=self.language_code)
        self.whisper_worker.text_recognized.connect(self.on_text_recognized)
        self.whisper_worker.command_detected.connect(self.on_command_detected)
        self.whisper_worker.start()

        # 重启录音
        self.audio_recorder = AudioRecorder(
            self.whisper_worker,
            samplerate=16000,
            block_duration=5.0,
            level_threshold=self.noise_threshold,
            parent=self,
        )
        self.audio_recorder.level_state_changed.connect(self.on_stt_level_state)
        self.audio_recorder.start()

    # ---------- 灵敏度 ----------
    def on_sensitivity_slider(self, value: int):
        alpha = value / 1000.0
        alpha = max(0.001, min(0.99, alpha))
        self.controller.filter.alpha = alpha
        self.sensitivity_edit.setText(f"{alpha:.3f}")

    def on_sensitivity_edit(self):
        text = self.sensitivity_edit.text().strip()
        try:
            alpha = float(text)
        except ValueError:
            alpha = self.controller.filter.alpha

        alpha = max(0.01, min(0.99, alpha))
        self.controller.filter.alpha = alpha
        self.sensitivity_slider.setValue(int(alpha * 100))
        self.sensitivity_edit.setText(f"{alpha:.2f}")

    # ---------- 噪声阈值 ----------
    def on_noise_slider(self, value: int):
        threshold = value / 1000.0  # 0.001 - 0.100
        threshold = max(0.001, min(0.1, threshold))
        self.noise_threshold = threshold
        self.noise_edit.setText(f"{threshold:.3f}")
        if self.audio_recorder is not None:
            self.audio_recorder.set_threshold(self.noise_threshold)

    def on_noise_edit(self):
        text = self.noise_edit.text().strip()
        try:
            threshold = float(text)
        except ValueError:
            threshold = self.noise_threshold

        threshold = max(0.001, min(0.1, threshold))
        self.noise_threshold = threshold
        self.noise_slider.setValue(int(threshold * 1000))
        self.noise_edit.setText(f"{threshold:.3f}")
        if self.audio_recorder is not None:
            self.audio_recorder.set_threshold(self.noise_threshold)

    # ---------- 置顶窗口 ----------
    def on_on_top_changed(self, state):
        on_top = (state == QtCore.Qt.Checked)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, on_top)
        self.show()

    # ---------- 语言切换 ----------
    def on_language_changed(self, index: int):
        self.language_code = "zh" if index == 0 else "en"
        # 更新 Whisper 识别语言
        if self.whisper_worker is not None:
            self.whisper_worker.set_language(self.language_code)
        # 更新 UI 文本
        self.retranslate_ui()

    # ---------- 关闭窗口 ----------
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.timer.stop()
        if self.audio_recorder is not None:
            self.audio_recorder.stop()
        if self.whisper_worker is not None:
            self.whisper_worker.stop()
            self.whisper_worker.wait()
        self.controller.release()
        event.accept()


# =========================
#   入口
# =========================

def main():
    pyautogui.FAILSAFE = False

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
