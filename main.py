import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel, QProgressBar, QMessageBox, QSlider, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

# Importujemy nasze moduły
from processing import AudioPlayer, calculate_energy_frames
from graphics_engine import FPSGLWidget, VideoHandler

CONTROL_PANEL_WIDTH_OPEN = 240
CONTROL_PANEL_WIDTH_CLOSED = 30

class CalculationWorker(QThread):
    finished = pyqtSignal(object, object, float, int, int)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    def __init__(self, filename, target_order, fps=30):
        super().__init__()
        self.filename = filename
        self.target_order = target_order
        self.fps = fps
    def run(self):
        try:
            # Używamy funkcji z processing.py
            xyz, energy, max_e, fs, n_frames = calculate_energy_frames(
                self.filename, self.progress.emit, target_order=self.target_order, fps=self.fps
            )
            self.finished.emit(xyz, energy, max_e, fs, n_frames)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Immersive Ambisonic Visualizer')
        self.resize(1200, 800)
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QHBoxLayout(cw)
        
        # Tworzymy widget OpenGL z graphics_engine.py
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
            QLabel { color: #aaaaaa; font-size: 24px; background: transparent; }
        """)
        self.placeholder_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placeholder_label.show()

        self.btn_collapse = QPushButton("▶")
        self.btn_collapse.setFixedWidth(24)
        self.btn_collapse.clicked.connect(self.toggle_menu)
        ctrl.addWidget(self.btn_collapse, alignment=Qt.AlignmentFlag.AlignRight)
        self.is_control_panel_open = True
        self.toggle_menu(force_open=True)

        lbl_order = QLabel("Ambisonic Order:")
        ctrl.addWidget(lbl_order)
        self.combo_order = QComboBox()
        self.combo_order.addItems(["Auto (Max)", "1st Order (4 ch)", "2nd Order (9 ch)", "3rd Order (16 ch)"])
        self.combo_order.currentIndexChanged.connect(self.on_order_changed)
        ctrl.addWidget(self.combo_order)

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
        ctrl.addSpacing(10)
        
        lbl_vol = QLabel("Gain (Boost):")
        ctrl.addWidget(lbl_vol)
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(20) 
        self.slider_volume.valueChanged.connect(self.update_volume)
        ctrl.addWidget(self.slider_volume)

        lbl_sharp = QLabel("Visual Focus:")
        ctrl.addWidget(lbl_sharp)
        self.slider_sharpness = QSlider(Qt.Orientation.Horizontal)
        self.slider_sharpness.setRange(1, 10)
        self.slider_sharpness.setValue(3)
        self.slider_sharpness.valueChanged.connect(self.update_sharpness)
        ctrl.addWidget(self.slider_sharpness)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        ctrl.addWidget(self.lbl_status)

        play_stop_layout = QHBoxLayout()
        play_stop_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.btn_play_stop = QPushButton("▶")
        self.btn_play_stop.setFixedSize(44, 44)
        self.btn_play_stop.setStyleSheet("QPushButton { border-radius: 22px; background: #2ecc71; font-size: 18px; }")
        self.btn_play_stop.clicked.connect(self.__manage_play_stop)
        ctrl.addWidget(self.btn_play_stop)
        play_stop_layout.addWidget(self.btn_play_stop)
        ctrl.addLayout(play_stop_layout)

        self.audio_progress = QProgressBar()
        self.audio_progress.setRange(0, 1000)
        self.audio_progress.setTextVisible(False)
        self.audio_progress.setFixedHeight(6)
        self.audio_progress.setStyleSheet("QProgressBar { background: #333; border-radius: 3px; } QProgressBar::chunk { background: #2ecc71; border-radius: 3px; }")
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
        self.update_sharpness(3)
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
                elif _layout: process_layout(_layout)
        process_layout(self.control_panel.layout())

    def update_sharpness(self, val): self.glw.set_sharpness(float(val))
    def update_volume(self, val):
        if self.audio_player: self.audio_player.set_gain(val)
    
    def update_audio_rotation(self, yaw, pitch):
        if self.audio_player:
            self.audio_player.set_view_rotation(yaw, pitch)

    def on_order_changed(self, idx):
        if self.audio_path:
            if self.is_playing: self.__manage_play_stop()
            self.start_processing(self.audio_path)

    def __manage_hint(self):
        is_show = self.audio_path is None or self.video_handler.image is None and not self.video_handler.is_video
        self.placeholder_label.setVisible(is_show)

    def __manage_play_stop(self):
        is_ready = self.audio_path is not None and (self.video_handler.image is not None or self.video_handler.is_video)
        if not is_ready: return None
        if not self.is_playing:
            if self.audio_player is None:
                self.audio_player = AudioPlayer(self.audio_path)
                self.audio_player.set_gain(self.slider_volume.value())
                self.audio_player.start()
            else: self.audio_player.resume()
            self.is_playing = True
            self.timer.start()
            self.btn_play_stop.setText("■")
        else:
            if self.audio_player is not None: self.audio_player.pause()
            self.is_playing = False
            self.timer.stop()
            self.btn_play_stop.setText("▶")

    def load_audio(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Audio', filter='WAV (*.wav)')
        if not fname: return
        self.start_processing(fname)

    def start_processing(self, fname):
        self.audio_path = fname
        self.lbl_status.setText("Processing Audio...")
        self.btn_audio.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_play_stop.setEnabled(False)
        self.combo_order.setEnabled(False)
        self.progress_bar.setValue(0)
        sel_idx = self.combo_order.currentIndex()
        target_order = None
        if sel_idx == 1: target_order = 1
        elif sel_idx == 2: target_order = 2
        elif sel_idx == 3: target_order = 3
        self.worker = CalculationWorker(fname, target_order, fps=self.fps)
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
        self.combo_order.setEnabled(True)
        self.progress_bar.setValue(0)

    def on_audio_ready(self, xyz, energy, max_e, fs, n_frames):
        self.energy_frames = energy
        self.total_frames = n_frames
        self.glw.max_energy = max_e
        self.glw.prepare_mesh_mapping(xyz)
        self.lbl_status.setText("Ready. Use headphones.")
        self.btn_audio.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_play_stop.setEnabled(True)
        self.combo_order.setEnabled(True)
        self.__manage_hint()

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', filter='Video (*.mp4 *.jpg *.png)')
        if not fname: return
        self.video_handler.load(fname)
        self.lbl_status.setText("Video Loaded.")
        self.__manage_hint()

    def update_loop(self):
        if not self.is_playing: return
        t = self.audio_player.get_current_time()
        self.lbl_time.setText(f"{int(t//60):02}:{t%60:05.2f}")
        if self.total_frames > 0:
            f_idx = int(t * self.fps) % self.total_frames
            if self.energy_frames is not None:
                self.glw.set_energy_data(self.energy_frames[f_idx])
        if self.video_handler.is_video:
            frame = self.video_handler.get_frame(t)
            self.glw.update_texture(frame)
        if self.audio_player:
            duration = self.total_frames / self.fps if self.total_frames else 1.0
            progress = int((t / duration) * 1000)
            self.audio_progress.setValue(min(progress, 1000))

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