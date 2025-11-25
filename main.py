import sys
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyfar as pf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel)
from PyQt6.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from matplotlib import cm

# --- Audio Utils ---
def to_stereo(data):
    if data.ndim == 1:
        return data
    if data.shape[1] == 2:
        return data
    left = np.mean(data[:, :data.shape[1]//2], axis=1)
    right = np.mean(data[:, data.shape[1]//2:], axis=1)
    return np.column_stack((left, right))

def calculate_energy_frames(wav_filename, n_points=4000, fps=30):
    """
    Calculates energy.
    n_points is now adjustable (passed from Worker).
    """
    signal = pf.io.read_audio(wav_filename)

    if signal.cshape != (4,):
        raise ValueError(f"Expected 4 channels (Ambisonics FOA), got {signal.cshape}.")

    fs = signal.sampling_rate
    total_samples = signal.n_samples
    hop_size = int(fs / fps)
    num_frames = int(total_samples / hop_size)

    raw_data = signal.time

    # Sampling the sphere
    sampling = pf.samplings.sph_equal_area(n_points)
    xyz_coords = sampling.get_cart(convention="right", unit="met")
    vx = xyz_coords[:, 0]
    vy = xyz_coords[:, 1]
    vz = xyz_coords[:, 2]

    energy_over_time = np.zeros((num_frames, n_points), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        end = start + hop_size
        if end > total_samples: break

        w_chunk = raw_data[0, start:end]
        x_chunk = raw_data[1, start:end]
        y_chunk = raw_data[2, start:end]
        z_chunk = raw_data[3, start:end]

        # Plane wave decomposition/projection onto sampling points
        P = (w_chunk[None, :] +
             x_chunk[None, :] * vx[:, None] +
             y_chunk[None, :] * vy[:, None] +
             z_chunk[None, :] * vz[:, None])

        frame_energy = np.sqrt(np.mean(P ** 2, axis=1))
        energy_over_time[i, :] = frame_energy

    max_energy = float(np.max(energy_over_time)) if energy_over_time.size > 0 else 1.0
    return vx, vy, vz, energy_over_time, max_energy, fs

# --- Worker Thread for Calculations ---
class CalculationWorker(QThread):
    finished = pyqtSignal(object, object, object, object, float, int)
    error = pyqtSignal(str)

    def __init__(self, filename, n_points=4000, fps=30):
        super().__init__()
        self.filename = filename
        self.n_points = n_points
        self.fps = fps

    def run(self):
        try:
            # Pass n_points to the calculation function
            vx, vy, vz, energy, max_e, fs = calculate_energy_frames(
                self.filename, n_points=self.n_points, fps=self.fps
            )
            self.finished.emit(vx, vy, vz, energy, max_e, fs)
        except Exception as e:
            self.error.emit(str(e))

# --- Audio Player Thread ---
class AudioPlayer(threading.Thread):
    def __init__(self, filename):
        super().__init__(daemon=True)
        self.filename = filename
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def run(self):
        data, fs = sf.read(self.filename, dtype='float32')
        data = to_stereo(data)
        blocksize = 1024
        num_samples = len(data)

        try:
            with sd.OutputStream(samplerate=fs, channels=2, blocksize=blocksize) as stream:
                start = 0
                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        sd.sleep(50)
                        continue

                    end = start + blocksize
                    if end >= num_samples:
                        block = data[start:num_samples]
                        if len(block) > 0: stream.write(block)
                        break

                    stream.write(data[start:end])
                    start = end
        except Exception as e:
            print("Audio playback error:", e)

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

# --- OpenGL Widget ---
class FPSGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vx = None
        self.vy = None
        self.vz = None
        self.energy_frames = None
        self.max_energy = 1.0
        self.frame_idx = 0

        # Camera State
        self.yaw = -90.0
        self.pitch = 0.0
        self.lastPos = QPoint()
        self.sensitivity = 0.2

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # [CHANGE 1] Reduced Point size so 4000 points don't look messy
        glPointSize(3.5)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # [CHANGE 2] Increased FOV from 60.0 to 100.0 for a wider view
        gluPerspective(100.0, w / max(1.0, h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        rad_yaw = np.radians(self.yaw)
        rad_pitch = np.radians(self.pitch)

        look_x = np.cos(rad_yaw) * np.cos(rad_pitch)
        look_y = np.sin(rad_pitch)
        look_z = np.sin(rad_yaw) * np.cos(rad_pitch)

        gluLookAt(0, 0, 0, look_x, look_y, look_z, 0, 1, 0)

        if self.energy_frames is None:
            return

        e_vals = self.energy_frames[self.frame_idx]
        normalized = np.clip(e_vals / (self.max_energy + 1e-12), 0.0, 1.0)

        cmap = cm.get_cmap('jet')
        rgba = cmap(normalized)[:, :3]

        glBegin(GL_POINTS)
        for i in range(len(self.vx)):
            glColor3f(float(rgba[i, 0]), float(rgba[i, 1]), float(rgba[i, 2]))
            r = 2.0
            glVertex3f(float(self.vx[i]*r), float(self.vy[i]*r), float(self.vz[i]*r))
        glEnd()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.lastPos.x()
        dy = event.pos().y() - self.lastPos.y()
        self.lastPos = event.pos()
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        if self.pitch > 89.0: self.pitch = 89.0
        if self.pitch < -89.0: self.pitch = -89.0
        self.update()

    def set_data(self, vx, vy, vz, energy_frames, max_energy):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.energy_frames = energy_frames
        self.max_energy = max_energy
        self.frame_idx = 0
        self.update()

    def set_frame(self, idx):
        if self.energy_frames is None: return
        self.frame_idx = int(idx) % len(self.energy_frames)
        self.update()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ambisonics Inside-View')
        self.resize(1100, 900)

        cw = QWidget()
        self.setCentralWidget(cw)
        hbox = QHBoxLayout(cw)

        self.glw = FPSGLWidget(self)
        hbox.addWidget(self.glw, stretch=1)

        ctrl = QVBoxLayout()

        self.load_btn = QPushButton('Load WAV')
        self.load_btn.clicked.connect(self.select_file)
        ctrl.addWidget(self.load_btn)

        self.status_label = QLabel("Please load a file.")
        self.status_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl.addWidget(self.status_label)

        self.start_btn = QPushButton('Start Experience')
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.toggle_start)
        ctrl.addWidget(self.start_btn)

        self.time_label = QLabel('t = 0.00 s')
        ctrl.addWidget(self.time_label)

        help_lbl = QLabel("\nControls:\nClick & Drag mouse\nto look around.")
        ctrl.addWidget(help_lbl)

        ctrl.addStretch(1)
        hbox.addLayout(ctrl)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.audio_thread = None
        self.worker = None
        self.loaded_filename = None
        self.energy_frames = None
        self.n_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.fps = 30

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open FOA WAV', filter='WAV Files (*.wav)')
        if not fname: return

        self.loaded_filename = fname

        self.status_label.setText("Calculating frames... Please wait.")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.load_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Start Experience")

        # [CHANGE 3] Increased n_points to 4000 (was 1000)
        self.worker = CalculationWorker(fname, n_points=4000, fps=self.fps)
        self.worker.finished.connect(self.on_calculations_done)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_calculations_done(self, vx, vy, vz, energy_frames, max_e, fs):
        self.energy_frames = energy_frames
        self.n_frames = len(energy_frames)
        self.current_frame = 0

        self.glw.set_data(vx, vy, vz, energy_frames, max_e)

        self.status_label.setText("Ready.")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.is_playing = False

    def on_worker_error(self, err_msg):
        self.status_label.setText(f"Error: {err_msg}")
        self.status_label.setStyleSheet("color: red;")
        self.load_btn.setEnabled(True)

    def toggle_start(self):
        if not self.is_playing and (self.audio_thread is None or not self.audio_thread.is_alive()):
            self.start_playback()
            self.start_btn.setText("Pause")
            self.is_playing = True
            return

        if self.is_playing:
            self.timer.stop()
            if self.audio_thread:
                self.audio_thread.pause()
            self.start_btn.setText("Resume")
            self.is_playing = False
        else:
            self.timer.start(int(1000 / self.fps))
            if self.audio_thread:
                self.audio_thread.resume()
            self.start_btn.setText("Pause")
            self.is_playing = True

    def start_playback(self):
        if self.loaded_filename is None: return

        self.current_frame = 0

        self.audio_thread = AudioPlayer(self.loaded_filename)
        self.audio_thread.start()

        self.timer.start(int(1000 / self.fps))

    def next_frame(self):
        if self.energy_frames is None: return

        self.current_frame += 1

        if self.current_frame >= self.n_frames:
            self.current_frame = 0

        self.glw.set_frame(self.current_frame)
        t = self.current_frame / max(1, self.fps)
        self.time_label.setText(f"t = {t:.2f} s")

    def closeEvent(self, event):
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.stop()
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        event.accept()

def main():
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()