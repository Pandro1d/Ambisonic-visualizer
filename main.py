import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel, QProgressBar, QMessageBox, QSlider, QComboBox
)
from PyQt6.QtCore import Qt, QTimer

# Importujemy nowe klasy
from processing import RealTimeAudioPlayer
from graphics_engine import FPSGLWidget, VideoHandler

CONTROL_PANEL_WIDTH_OPEN = 240
CONTROL_PANEL_WIDTH_CLOSED = 30

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Real-Time Ambisonic Visualizer')
        self.resize(1200, 800)
        
        # --- UI SETUP ---
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
        
        self.placeholder_label = QLabel("Load Audio/Video to start", self.glw)
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
        ctrl.addWidget(self.combo_order)

        self.btn_audio = QPushButton('1. Load Audio')
        self.btn_audio.clicked.connect(self.load_audio)
        ctrl.addWidget(self.btn_audio)

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

        self.lbl_time = QLabel("00:00")
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch()

        # --- STATE ---
        self.audio_path = None
        self.player = None
        self.video_handler = VideoHandler()
        self.is_playing = False
        self.update_sharpness(3)
        
        # Timer do wideo (audio odświeża się samo przez sygnały)
        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_video_frame)

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
        if self.player: self.player.set_gain(val)
    
    def update_audio_rotation(self, yaw, pitch):
        if self.player: self.player.set_view_rotation(yaw, pitch)

    def load_audio(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Audio', filter='WAV (*.wav)')
        if not fname: return
        
        # Stop existing
        if self.player: 
            self.player.stop()
            self.player.wait()
            
        self.audio_path = fname
        self.lbl_status.setText("Audio Selected. Press Play.")
        self.btn_video.setEnabled(True)
        self.btn_play_stop.setEnabled(True)
        self.__manage_hint()
        
        # Reset play state
        self.is_playing = False
        self.btn_play_stop.setText("▶")

    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', filter='Video (*.mp4 *.jpg *.png)')
        if not fname: return
        self.video_handler.load(fname)
        self.lbl_status.setText("Video Loaded.")
        self.__manage_hint()
        # Odśwież jedną klatkę, żeby pokazać wideo od razu
        self.update_video_frame(force=True)

    def __manage_hint(self):
        is_show = self.audio_path is None
        self.placeholder_label.setVisible(is_show)

    def __manage_play_stop(self):
        if not self.audio_path: return

        if not self.is_playing:
            # START PLAYBACK
            if self.player is None or not self.player.isRunning():
                # Wybór rzędu
                sel_idx = self.combo_order.currentIndex()
                target_order = None
                if sel_idx == 1: target_order = 1
                elif sel_idx == 2: target_order = 2
                elif sel_idx == 3: target_order = 3

                # Inicjalizacja nowego odtwarzacza
                self.player = RealTimeAudioPlayer(self.audio_path, target_order)
                
                # Podłączenie sygnałów
                self.player.mesh_initialized.connect(self.glw.prepare_mesh_mapping)
                self.player.energy_updated.connect(self.glw.set_energy_data)
                self.player.error_occurred.connect(self.on_error)
                self.player.finished_playback.connect(self.on_finished)
                
                # Ustawienia początkowe
                self.player.set_gain(self.slider_volume.value())
                
                # --- KLUCZOWA POPRAWKA: Synchronizacja kąta przy starcie ---
                # Pobieramy aktualny kąt z widgetu graficznego i wysyłamy do audio
                # zanim zacznie grać. Dzięki temu audio "wie", gdzie patrzysz.
                self.player.set_view_rotation(self.glw.yaw, self.glw.pitch)
                
                self.player.start()
            else:
                self.player.resume()
            
            self.is_playing = True
            self.timer.start()
            self.btn_play_stop.setText("■")
            self.lbl_status.setText("Playing...")
        else:
            # PAUSE
            if self.player: self.player.pause()
            self.is_playing = False
            self.timer.stop()
            self.btn_play_stop.setText("▶")
            self.lbl_status.setText("Paused")

    def update_video_frame(self, force=False):
        if not self.player and not force: return
        
        # Pobierz czas z audio (master clock)
        t = 0.0
        if self.player:
            t = self.player.get_current_time()
        
        self.lbl_time.setText(f"{int(t//60):02}:{t%60:05.2f}")

        # Aktualizuj wideo
        if self.video_handler.is_video or force:
            frame = self.video_handler.get_frame(t)
            self.glw.update_texture(frame)
        elif self.video_handler.image is not None and force:
             self.glw.update_texture(self.video_handler.image)

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.__manage_play_stop() # Reset buttons

    def on_finished(self):
        self.is_playing = False
        self.timer.stop()
        self.btn_play_stop.setText("▶")
        self.lbl_status.setText("Finished")

    def closeEvent(self, event):
        if self.player: 
            self.player.stop()
            self.player.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()