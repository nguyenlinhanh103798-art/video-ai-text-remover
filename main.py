
import sys
import os
import cv2
import numpy as np
import torch
import easyocr
import subprocess
import imageio_ffmpeg

from pathlib import Path
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QProgressBar,
    QTextEdit, QMessageBox, QRubberBand
)


class VideoPreview(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#222;color:white;")
        self.setText("Chưa chọn video")
        self.rubber = QRubberBand(
            QRubberBand.Shape.Rectangle, self
        )
        self.origin = None
        self.selected_rect = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.rubber.setGeometry(
                self.origin.x(), self.origin.y(), 0, 0
            )
            self.rubber.show()

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            rect = self.origin.__class__(
                event.position().toPoint()
            )
            from PySide6.QtCore import QRect
            self.rubber.setGeometry(
                QRect(self.origin, rect).normalized()
            )

    def mouseReleaseEvent(self, event):
        if self.origin is not None:
            self.selected_rect = self.rubber.geometry()
            self.origin = None


class VideoWorker(QThread):
    progress = Signal(int)
    message = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, input_path, output_path, manual_rect):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.manual_rect = manual_rect

    def run(self):
        cap = None
        writer = None

        try:
            self.message.emit("Đang khởi tạo AI nhận diện chữ...")

            reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=torch.cuda.is_available()
            )

            cap = cv2.VideoCapture(self.input_path)

            if not cap.isOpened():
                raise Exception("Không mở được video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise Exception("Kích thước video không hợp lệ.")

            silent_path = str(
                Path(self.output_path).with_name(
                    Path(self.output_path).stem + "_silent.mp4"
                )
            )

            writer = cv2.VideoWriter(
                silent_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                raise Exception("Không tạo được video đầu ra.")

            manual_mask = np.zeros((height, width), dtype=np.uint8)

            if self.manual_rect:
                x1, y1, x2, y2 = self.manual_rect
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))

                if x2 > x1 and y2 > y1:
                    manual_mask[y1:y2, x1:x2] = 255

            frame_number = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                results = reader.readtext(
                    frame,
                    detail=1,
                    paragraph=False
                )

                mask = manual_mask.copy()

                for result in results:
                    box = np.array(
                        result[0], dtype=np.int32
                    )
                    cv2.fillPoly(mask, [box], 255)

                # Mở rộng vùng che chữ để tránh viền còn sót.
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)

                # Phục hồi vùng ảnh bị chữ che.
                cleaned = cv2.inpaint(
                    frame, mask, 5, cv2.INPAINT_TELEA
                )

                writer.write(cleaned)
                frame_number += 1

                if total > 0:
                    percent = int(frame_number * 100 / total)
                    self.progress.emit(min(percent, 100))

                if frame_number % 30 == 0:
                    self.message.emit(
                        f"Đang xử lý khung hình {frame_number}/{total}"
                    )

            cap.release()
            cap = None
            writer.release()
            writer = None

            self.message.emit("Đang ghép lại âm thanh gốc...")

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            command = [
                ffmpeg, "-y",
                "-i", silent_path,
                "-i", self.input_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                self.output_path
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                os.replace(silent_path, self.output_path)
                self.message.emit(
                    "Không ghép được âm thanh; đã xuất video không tiếng."
                )
            else:
                os.remove(silent_path)

            self.progress.emit(100)
            self.finished_ok.emit(self.output_path)

        except Exception as error:
            self.failed.emit(str(error))

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Video Text Remover")
        self.resize(850, 700)

        self.input_path = ""
        self.output_path = ""

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        title = QLabel("AI XÓA CHỮ TRONG VIDEO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:22px;font-weight:bold;padding:12px;"
        )
        layout.addWidget(title)

        self.preview = VideoPreview()
        layout.addWidget(
            self.preview, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.btn_input = QPushButton("1. CHỌN VIDEO")
        self.btn_input.clicked.connect(self.choose_video)
        layout.addWidget(self.btn_input)

        self.file_label = QLabel("Chưa chọn video")
        layout.addWidget(self.file_label)

        self.btn_output = QPushButton("2. CHỌN NƠI LƯU VIDEO")
        self.btn_output.clicked.connect(self.choose_output)
        layout.addWidget(self.btn_output)

        self.output_label = QLabel("Chưa chọn nơi lưu")
        layout.addWidget(self.output_label)

        self.btn_clear = QPushButton("XÓA VÙNG CHỌN")
        self.btn_clear.clicked.connect(self.clear_region)
        layout.addWidget(self.btn_clear)

        note = QLabel(
            "Có thể kéo chuột trên hình xem trước để chọn thêm vùng chữ."
        )
        layout.addWidget(note)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

        self.btn_start = QPushButton("3. BẮT ĐẦU XÓA CHỮ BẰNG AI")
        self.btn_start.clicked.connect(self.start_process)
        self.btn_start.setStyleSheet(
            "background:#168b45;color:white;"
            "font-weight:bold;padding:12px;"
        )
        layout.addWidget(self.btn_start)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video (*.mp4 *.mov *.mkv *.avi)"
        )

        if not path:
            return

        self.input_path = path
        self.file_label.setText(path)

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()

        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimage = QImage(
                rgb.data, w, h, ch * w,
                QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            self.preview.setPixmap(
                pixmap.scaled(
                    640, 360,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.preview.selected_rect = None

        default_output = str(
            Path(path).with_name(
                Path(path).stem + "_clean.mp4"
            )
        )
        self.output_path = default_output
        self.output_label.setText(default_output)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu video đã xử lý",
            self.output_path or "video_clean.mp4",
            "MP4 Video (*.mp4)"
        )

        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_path = path
            self.output_label.setText(path)

    def clear_region(self):
        self.preview.selected_rect = None
        self.preview.rubber.hide()
        self.log.append("Đã xóa vùng chọn thủ công.")

    def start_process(self):
        if not self.input_path:
            QMessageBox.warning(
                self, "Thông báo", "Bạn chưa chọn video."
            )
            return

        if not self.output_path:
            QMessageBox.warning(
                self, "Thông báo", "Bạn chưa chọn nơi lưu."
            )
            return

        manual_rect = None
        rect = self.preview.selected_rect

        if rect and not rect.isNull():
            cap = cv2.VideoCapture(self.input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            x1 = int(rect.left() * width / 640)
            y1 = int(rect.top() * height / 360)
            x2 = int(rect.right() * width / 640)
            y2 = int(rect.bottom() * height / 360)

            manual_rect = (x1, y1, x2, y2)

        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.log.append("Bắt đầu xử lý video...")

        self.worker = VideoWorker(
            self.input_path,
            self.output_path,
            manual_rect
        )

        self.worker.progress.connect(self.progress.setValue)
        self.worker.message.connect(self.log.append)
        self.worker.finished_ok.connect(self.process_done)
        self.worker.failed.connect(self.process_error)
        self.worker.start()

    def process_done(self, path):
        self.btn_start.setEnabled(True)
        self.log.append("HOÀN THÀNH: " + path)
        QMessageBox.information(
            self,
            "Hoàn thành",
            "Video đã được xử lý và lưu tại:\n" + path
        )

    def process_error(self, error):
        self.btn_start.setEnabled(True)
        self.log.append("LỖI: " + error)
        QMessageBox.critical(self, "Lỗi xử lý", error)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
