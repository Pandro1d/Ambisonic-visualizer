import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyfar as pf
import scipy.special as sp
import warnings
from PyQt6.QtCore import QThread, pyqtSignal

# Wyciszenie ostrzeżeń Pyfar
warnings.filterwarnings("ignore", category=UserWarning, module='pyfar')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pyfar')

def get_sh_matrix(coords_sph, max_order):
    """Generuje macierz transformacji dla Harmonik Sferycznych."""
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

class RealTimeAudioPlayer(QThread):
    # Sygnały
    mesh_initialized = pyqtSignal(object) 
    energy_updated = pyqtSignal(object)
    finished_playback = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, filename, target_order=None):
        super().__init__()
        self.filename = filename
        self.target_order = target_order
        self._stop_event = False
        self._pause_event = False
        self.start_time = 0
        self.pause_time = 0
        self.is_running = False
        
        # Parametry
        self.gain = 1.0 
        self.current_yaw = -90.0
        self.current_pitch = 0.0
        self.n_points_viz = 1024 

    def set_gain(self, vol_slider_val):
        self.gain = (vol_slider_val / 100.0) * 5.0

    def set_view_rotation(self, yaw, pitch):
        self.current_yaw = yaw
        self.current_pitch = pitch

    def stop(self): self._stop_event = True
    def pause(self): self._pause_event = True
    def resume(self):
        self._pause_event = False
        self.start_time = time.time() - self.pause_time

    def get_current_time(self):
        if self._pause_event: return self.pause_time
        if not self.is_running: return 0.0
        return max(0.0, time.time() - self.start_time)

    def run(self):
        try:
            # 1. Wczytywanie pliku
            f = sf.SoundFile(self.filename)
            fs = f.samplerate
            n_channels = f.channels
            total_samples = len(f)
            
            # Walidacja
            max_possible_order = int(np.sqrt(n_channels) - 1)
            if max_possible_order < 1 and n_channels < 4:
                self.error_occurred.emit(f"Not enough channels (min 4). Found: {n_channels}")
                return

            if self.target_order is None or self.target_order > max_possible_order:
                order = max_possible_order
            else:
                order = self.target_order

            channels_used = (order + 1) ** 2
            
            # 2. Inicjalizacja Matematyki
            sampling = pf.samplings.sph_equal_area(self.n_points_viz, radius=5.0)
            xyz_coords = sampling.cartesian # Oryginał: [X, Y, Z] gdzie Z to góra
            sph_coords = np.column_stack((sampling.azimuth, sampling.elevation, sampling.radius))
            
            # --- NAPRAWA OSII WIZUALNYCH ---
            # PyFar: X=Przód, Y=Lewo, Z=Góra
            # OpenGL: X=Lewo, Y=Góra, Z=Przód
            # Musimy zamienić kolorymacji, aby Audio Z (Góra) trafiło do OpenGL Y (Góra)
            
            # Nowy układ dla grafiki: [Audio_Y, Audio_Z, Audio_X]
            # Dzięki temu:
            # - OpenGL X (poziom) dostaje Audio Y (lewo/prawo) -> OK
            # - OpenGL Y (pion)   dostaje Audio Z (góra/dół)   -> NAPRAWA PROBLEMU
            # - OpenGL Z (głębia) dostaje Audio X (przód/tył)  -> OK
            
            # Dodatkowo negujemy X (Audio_Y), aby dopasować kierunek obrotu wideo
            xyz_visuals = np.column_stack((
                -xyz_coords[:, 1], # Audio Y -> GL X (Lewo/Prawo)
                xyz_coords[:, 2],  # Audio Z -> GL Y (Góra/Dół) - TO JEST KLUCZOWE
                xyz_coords[:, 0]   # Audio X -> GL Z (Przód/Tył)
            ))
            
            # Generujemy macierz SH na podstawie ORYGINALNYCH współrzędnych sferycznych (matematyka się zgadza)
            Y_matrix = get_sh_matrix(sph_coords, order)
            
            # Wysyłamy PRZEMAPOWANE koordynaty do wizualizacji
            self.mesh_initialized.emit(xyz_visuals)

            # 3. Pętla przetwarzania
            blocksize = 1024 
            
            with sd.OutputStream(samplerate=fs, channels=2, blocksize=blocksize, dtype='float32') as stream:
                self.is_running = True
                self.start_time = time.time()

                for block in f.blocks(blocksize=blocksize, dtype='float32', always_2d=True):
                    if self._stop_event: break
                    
                    while self._pause_event:
                        sd.sleep(10) # 10ms (int)
                        self.start_time = time.time() - self.pause_time
                        if self._stop_event: break
                    
                    current_pos = f.tell()
                    self.pause_time = (current_pos / fs)
                    if current_pos < blocksize * 2: self.start_time = time.time()

                    # --- A. WIZUALIZACJA ---
                    ambi_chunk = block[:, :channels_used]
                    pressure = np.dot(ambi_chunk, Y_matrix.T)
                    energy_snapshot = np.sqrt(np.mean(pressure ** 2, axis=0))
                    self.energy_updated.emit(energy_snapshot)

                    # --- B. DŹWIĘK ---
                    if n_channels >= 4:
                        w = block[:, 0]
                        y = block[:, 1]
                        z = block[:, 2]
                        x = block[:, 3]

                        theta = np.radians((self.current_yaw + 90))
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)

                        x_rot = x * cos_t - y * sin_t
                        y_rot = x * sin_t + y * cos_t
                        
                        sd_coef = 0.7071
                        left = 0.5 * w + 0.5 * (sd_coef * x_rot + sd_coef * y_rot)
                        right = 0.5 * w + 0.5 * (sd_coef * x_rot - sd_coef * y_rot)
                        out_stereo = np.column_stack((left, right))
                    elif n_channels == 2:
                        out_stereo = block
                    else:
                        out_stereo = np.column_stack((block, block))

                    out_stereo = out_stereo * self.gain
                    np.clip(out_stereo, -1.0, 1.0, out=out_stereo)
                    
                    stream.write(out_stereo.astype(np.float32))

            self.finished_playback.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))
            print(f"Error: {e}")
        finally:
            self.is_running = False