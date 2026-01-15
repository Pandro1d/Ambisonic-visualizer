import sys
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyfar as pf
import scipy.special as sp
from scipy.spatial import cKDTree
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from matplotlib import cm


CONTROL_PANEL_WIDTH_OPEN = 200
CONTROL_PANEL_WIDTH_CLOSED = 30

# --- Helper: Generate Efficient Sphere Mesh ---
def create_sphere_mesh_fast(radius, slices, stacks):
    vertices = []
    uvs = []

    for i in range(stacks + 1):
        lat = np.pi * (-0.5 + float(i) / stacks)
        z = np.sin(lat)
        zr = np.cos(lat)

        for j in range(slices + 1):
            lng = 2 * np.pi * float(j) / slices
            x = zr * np.cos(lng)
            y = zr * np.sin(lng)

            vertices.append([x * radius, z * radius, y * radius])

            # Flip V coordinate for correct texture mapping
            u = float(j) / slices
            v = 1.0 - (float(i) / stacks)
            uvs.append([u, v])

    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    indices = []
    stride = slices + 1
    for i in range(stacks):
        for j in range(slices):
            p1 = i * stride + j
            p2 = p1 + 1
            p3 = (i + 1) * stride + j + 1
            p4 = (i + 1) * stride + j
            indices.extend([p1, p2, p3, p4])

    indices = np.array(indices, dtype=np.uint32)
    return vertices, uvs, indices

# --- Audio Math ---
def get_sh_matrix(coords_sph, max_order):
    n_points = coords_sph.shape[0]
    n_channels = (max_order + 1) ** 2
    Y = np.zeros((n_points, n_channels), dtype=np.float32)

    azimuth = coords_sph[:, 0]
    colatitude = np.pi/2 - coords_sph[:, 1]

    idx = 0
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            y_c = sp.sph_harm(m, n, azimuth, colatitude)
            if m == 0: y_real = np.real(y_c)
            elif m > 0: y_real = np.sqrt(2) * np.real(y_c)
            else:
                y_c_pos = sp.sph_harm(abs(m), n, azimuth, colatitude)
                y_real = np.sqrt(2) * np.imag(y_c_pos) * (-1 if (abs(m) % 2) == 1 else 1)
            norm_factor = np.sqrt(4 * np.pi) / np.sqrt(2 * n + 1)
            Y[:, idx] = y_real * norm_factor
            idx += 1
    return Y

def calculate_energy_frames(wav_filename, progress_callback, n_points=4000, fps=30):
    signal = pf.io.read_audio(wav_filename)
    n_channels = signal.cshape[0]

    # --- VALIDATION: Check for valid Ambisonic Order ---
    calc_order = np.sqrt(n_channels) - 1
    if not calc_order.is_integer() or n_channels < 4:
        raise ValueError(f"Invalid Channel Count: {n_channels}.\nAmbisonics requires square channel counts (4, 9, 16, 25...).")

    order = int(calc_order)
    print(f"Detected Ambisonic Order: {order} ({n_channels} channels)")

    fs = signal.sampling_rate
    hop_size = int(fs / fps)
    num_frames = int(signal.n_samples / hop_size)
    raw_data = signal.time

    sampling = pf.samplings.sph_equal_area(n_points)
    xyz_coords = sampling.cartesian
    sph_coords = np.column_stack((sampling.azimuth, sampling.elevation, sampling.radius))

    Y_matrix = get_sh_matrix(sph_coords, order)

    energy_over_time = np.zeros((num_frames, n_points), dtype=np.float32)

    for i in range(num_frames):
        if i % 10 == 0:
            percent = int((i / num_frames) * 100)
            progress_callback(percent)

        start = i * hop_size
        end = start + hop_size
        if end > signal.n_samples: break

        chunk = raw_data[:, start:end].T
        pressure = np.dot(chunk, Y_matrix.T)
        energy_over_time[i, :] = np.sqrt(np.mean(pressure ** 2, axis=0))

    progress_callback(100)
    max_energy = float(np.max(energy_over_time)) if energy_over_time.size > 0 else 1.0
    return xyz_coords, energy_over_time, max_energy, fs, num_frames

# --- Threads ---
class CalculationWorker(QThread):
    finished = pyqtSignal(object, object, float, int, int)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, filename, fps=30):
        super().__init__()
        self.filename = filename
        self.fps = fps

    def run(self):
        try:
            xyz, energy, max_e, fs, n_frames = calculate_energy_frames(
                self.filename, self.progress.emit, fps=self.fps
            )
            self.finished.emit(xyz, energy, max_e, fs, n_frames)
        except Exception as e:
            self.error.emit(str(e))

class AudioPlayer(threading.Thread):
    def __init__(self, filename):
        super().__init__(daemon=True)
        self.filename = filename
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.start_time = 0
        self.pause_time = 0
        self.is_running = False

    def run(self):
        # Read the full multi-channel audio
        data, fs = sf.read(self.filename, dtype='float32')
        n_channels = data.shape[1] if data.ndim > 1 else 1
        total_samples = len(data)

        stereo = None

        # --- DECODING LOGIC (Binaural / Spatial Stereo) ---
        if n_channels >= 4:
            # W (Omni) + Y (Side) -> Stereo
            w = data[:, 0]
            y = data[:, 1]
            left = 0.5 * (w + y)
            right = 0.5 * (w - y)
            stereo = np.column_stack((left, right))
        elif n_channels == 2:
            stereo = data
        elif n_channels == 1:
            stereo = np.column_stack((data, data))
        else:
            stereo = data[:, :2]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(stereo))
        if max_val > 1.0:
            stereo /= max_val

        blocksize = 2048
        idx = 0

        try:
            with sd.OutputStream(samplerate=fs, channels=2, blocksize=blocksize) as stream:
                self.is_running = True
                self.start_time = time.time()

                # LOOP FOREVER until stopped
                while not self._stop_event.is_set():

                    if self._pause_event.is_set():
                        sd.sleep(10)
                        # Keep start_time fresh so resuming doesn't jump
                        self.start_time = time.time() - self.pause_time
                        continue

                    # Update playback position for the GUI
                    self.pause_time = (idx / fs)

                    # --- GAPLESS LOOPING LOGIC ---
                    end = idx + blocksize

                    if end <= total_samples:
                        # Normal Case: We have enough data
                        chunk = stereo[idx:end]
                        idx = end
                    else:
                        # Wrap Case: We hit the end, take remainder + start of file
                        remainder = total_samples - idx
                        part1 = stereo[idx:]
                        part2 = stereo[:blocksize - remainder] # Wrap to start
                        chunk = np.concatenate((part1, part2))

                        # Reset for next loop
                        idx = blocksize - remainder

                        # Reset the synchronization clock for the Video
                        self.start_time = time.time() - (idx / fs)

                    stream.write(chunk)

        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.is_running = False

    def get_current_time(self):
        if self._pause_event.is_set(): return self.pause_time
        if not self.is_running: return 0.0
        # This calc relies on self.start_time being reset in the loop above
        t = time.time() - self.start_time
        return max(0.0, t)

    def stop(self): self._stop_event.set()
    def pause(self): self._pause_event.set()
    def resume(self):
        self._pause_event.clear()
        self.start_time = time.time() - self.pause_time

# --- Video Handler ---
class VideoHandler:
    def __init__(self):
        self.cap = None
        self.image = None
        self.is_video = False
        self.fps = 0
        self.frame_count = 0
        self.last_frame = None

    def load(self, path):
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_video = True
            return True
        img = cv2.imread(path)
        if img is not None:
            self.is_video = False
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True
        return False

    def get_frame(self, t_sec):
        if not self.is_video: return self.image

        target = int(t_sec * self.fps)

        # Looping Logic for Video:
        # If the audio looped, t_sec is small again.
        # If t_sec asks for a frame beyond video length, wrap it.
        if target >= self.frame_count:
            target = target % self.frame_count

        curr = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Seek only if we drifted significantly
        if abs(curr - target) > 5:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, frame = self.cap.read()
        if ret:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.last_frame

        # If read failed (end of file), seek to 0 and try again
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.last_frame

        return self.last_frame

# --- OpenGL Visualizer ---
class FPSGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.energy_frames = None
        self.max_energy = 1.0
        self.slices = 60
        self.stacks = 40
        self.verts = None
        self.uvs = None
        self.indices = None
        self.vertex_indices = None
        self.colors = None
        self.yaw = -90.0
        self.pitch = 0.0
        self.lastPos = QPoint()
        self.sensitivity = 0.2
        self.fov = 100.0
        self.texture_id = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        self.verts, self.uvs, self.indices = create_sphere_mesh_fast(50.0, self.slices, self.stacks)

    def prepare_mesh_mapping(self, audio_xyz_coords):
        norms = np.linalg.norm(self.verts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_mesh = self.verts / norms
        tree = cKDTree(audio_xyz_coords)
        _, self.vertex_indices = tree.query(unit_mesh)

    def update_texture(self, img_data):
        if img_data is None: return
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        h, w, c = img_data.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    def set_energy_data(self, energy_array):
        if self.vertex_indices is None: return
        mesh_energies = energy_array[self.vertex_indices]
        norm_e = np.clip(mesh_energies / (self.max_energy + 1e-9), 0.0, 1.0)
        cmap = cm.get_cmap('jet')
        rgba = cmap(norm_e)
        rgba[:, 3] = rgba[:, 3] * norm_e * 0.9
        self.colors = np.ascontiguousarray(rgba, dtype=np.float32)
        self.update()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, w / max(1.0, h), 0.1, 100.0)
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
        if self.verts is None: return
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.verts)
        glTexCoordPointer(2, GL_FLOAT, 0, self.uvs)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor3f(0.35, 0.35, 0.35)
        glDrawElements(GL_QUADS, len(self.indices), GL_UNSIGNED_INT, self.indices)
        glDisable(GL_TEXTURE_2D)

        if self.colors is not None:
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glEnableClientState(GL_COLOR_ARRAY)
            glPushMatrix()
            glScalef(0.95, 0.95, 0.95)
            glColorPointer(4, GL_FLOAT, 0, self.colors)
            glDrawElements(GL_QUADS, len(self.indices), GL_UNSIGNED_INT, self.indices)
            glPopMatrix()
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def mousePressEvent(self, event): self.lastPos = event.pos()
    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.lastPos.x()
        dy = event.pos().y() - self.lastPos.y()
        self.lastPos = event.pos()
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.update()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ambisonic visualizer')
        self.resize(1200, 800)

        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QHBoxLayout(cw)
        self.glw = FPSGLWidget(self)
        layout.addWidget(self.glw)

        self.control_panel = QWidget()

        ctrl = QVBoxLayout(self.control_panel)
        ctrl.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.glw)
        layout.addWidget(self.control_panel)
        self.placeholder_label = QLabel("Select files in menu", self.glw)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 24px;
                background: transparent;
            }
        """)
        self.placeholder_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placeholder_label.show()


        self.btn_collapse = QPushButton("▶")
        self.btn_collapse.setFixedWidth(24)
        self.btn_collapse.clicked.connect(self.toggle_menu)
        ctrl.addWidget(self.btn_collapse, alignment=Qt.AlignmentFlag.AlignRight)

        self.is_control_panel_open = True
        self.toggle_menu(force_open=True)

        self.btn_audio = QPushButton('1. Load Audio')
        self.btn_audio.clicked.connect(self.load_audio)
        ctrl.addWidget(self.btn_audio)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        ctrl.addWidget(self.progress_bar)

        self.btn_video = QPushButton('2. Load Video')
        self.btn_video.clicked.connect(self.load_video)
        self.btn_video.setEnabled(False)
        ctrl.addWidget(self.btn_video)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        ctrl.addWidget(self.lbl_status)

        play_stop_layout = QHBoxLayout()
        play_stop_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.btn_play_stop = QPushButton("▶")
        self.btn_play_stop.setFixedSize(44, 44)
        self.btn_play_stop.setStyleSheet("""
            QPushButton {
                border-radius: 22px;
                background: #2ecc71;
                font-size: 18px;
            }
        """)
        self.btn_play_stop.clicked.connect(self.__manage_play_stop)
        ctrl.addWidget(self.btn_play_stop)
        play_stop_layout.addWidget(self.btn_play_stop)
        ctrl.addLayout(play_stop_layout)

        self.audio_progress = QProgressBar()
        self.audio_progress.setRange(0, 1000)
        self.audio_progress.setTextVisible(False)
        self.audio_progress.setFixedHeight(6)
        self.audio_progress.setStyleSheet("""
            QProgressBar {
                background: #333;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #2ecc71;
                border-radius: 3px;
            }
        """)
        ctrl.addWidget(self.audio_progress)



        self.lbl_time = QLabel("00:00")
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch()

        self.audio_path = None
        self.worker = None
        self.audio_player = None
        self.video_handler = VideoHandler()
        self.energy_frames = None
        self.fps = 30
        self.is_playing = False
        self.total_frames = 0

        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_loop)

    def toggle_menu(self, force_open=False):
        is_openning = force_open or self.is_control_panel_open is False
        self.control_panel.setMaximumWidth(CONTROL_PANEL_WIDTH_OPEN if is_openning else CONTROL_PANEL_WIDTH_CLOSED)
        self.control_panel.setMinimumWidth(CONTROL_PANEL_WIDTH_OPEN if is_openning else CONTROL_PANEL_WIDTH_CLOSED)
        self.btn_collapse.setText("▶" if is_openning else "◀")
        self.__show_or_hide_menu(not is_openning)
        self.is_control_panel_open = is_openning

    def __show_or_hide_menu(self, hide):
        def process_layout(layout):
            for idx in range(layout.count()):
                item = layout.itemAt(idx)
                _widget = item.widget()
                _layout = item.layout()

                if _widget not in [None, self.btn_collapse]:
                    _widget.setVisible(not hide)
                elif _layout:
                    process_layout(_layout)
        process_layout(self.control_panel.layout())


    def __manage_hint(self):
        is_show = self.audio_path is None or self.video_handler.image is None and not self.video_handler.is_video
        self.placeholder_label.setVisible(is_show)

    def __manage_play_stop(self):
        is_ready = self.audio_path is not None and (self.video_handler.image is not None or self.video_handler.is_video)
        if not is_ready:
            return None

        if not self.is_playing:
            if self.audio_player is None:
                self.audio_player = AudioPlayer(self.audio_path)
                self.audio_player.start()
            else:
                self.audio_player.resume()
            self.is_playing = True
            self.timer.start()
            self.btn_play_stop.setText("■")
        else:
            if self.audio_player is not None:
                self.audio_player.pause()
            self.is_playing = False
            self.timer.stop()
            self.btn_play_stop.setText("▶")
            self.audio_progress.setValue(int((self.audio_player.pause_time / (self.total_frames / self.fps)) * 1000) if self.audio_player and self.total_frames else 0)

    def load_audio(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Audio', filter='WAV (*.wav)')
        if not fname: return
        self.audio_path = fname
        self.lbl_status.setText("Processing Audio...")
        self.btn_audio.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_play_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self.worker = CalculationWorker(fname, fps=self.fps)
        self.worker.finished.connect(self.on_audio_ready)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self.__manage_hint()

    def update_progress(self, val): self.progress_bar.setValue(val)
    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.lbl_status.setText("Error occurred.")
        self.btn_audio.setEnabled(True)
        self.progress_bar.setValue(0)

    def on_audio_ready(self, xyz, energy, max_e, fs, n_frames):
        self.energy_frames = energy
        self.total_frames = n_frames
        self.glw.max_energy = max_e
        self.glw.prepare_mesh_mapping(xyz)
        self.lbl_status.setText("Audio Processed. You may now load a video.")
        self.btn_audio.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_play_stop.setEnabled(True)
        self.__manage_hint()

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', filter='Video (*.mp4 *.jpg *.png)')
        if not fname: return
        self.video_handler.load(fname)
        self.lbl_status.setText("Video Loaded.")
        self.__manage_hint()

    def toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            if self.audio_player: self.audio_player.pause()
            self.btn_play_stop.setText("Play")
        else:
            if not self.audio_path: return
            if not self.audio_player:
                self.audio_player = AudioPlayer(self.audio_path)
                self.audio_player.start()
            else:
                self.audio_player.resume()
            self.is_playing = True
            self.timer.start()
            self.btn_play_stop.setText("Pause")

    def update_loop(self):
        if not self.is_playing: return
        t = self.audio_player.get_current_time()
        self.lbl_time.setText(f"{int(t//60):02}:{t%60:05.2f}")

        # Cycle the heatmap frame index
        if self.total_frames > 0:
            f_idx = int(t * self.fps) % self.total_frames
            if self.energy_frames is not None:
                self.glw.set_energy_data(self.energy_frames[f_idx])

        if self.video_handler.is_video:
            frame = self.video_handler.get_frame(t)
            self.glw.update_texture(frame)

        if self.audio_player:
            t = self.audio_player.get_current_time()
            duration = self.total_frames / self.fps if self.total_frames else 1.0
            progress = int((t / duration) * 1000)
            self.audio_progress.setValue(min(progress, 1000))
        else:
            self.audio_progress.setValue(0)


    def closeEvent(self, event):
        if self.audio_player: self.audio_player.stop()
        if self.worker: self.worker.terminate()
        self.is_playing = False
        event.accept()

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
