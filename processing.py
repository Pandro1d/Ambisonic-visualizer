import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyfar as pf
import scipy.special as sp

# --- Audio Math for Visualization ---
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

def calculate_energy_frames(wav_filename, progress_callback, target_order=None, n_points=1024, fps=30):
    """
    Zoptymalizowana wersja z przetwarzaniem wektorowym (batch processing).
    Zmniejszono domyślne n_points z 4000 na 1024 (wystarczające dla wizualizacji).
    """
    # 1. Wczytaj audio (tylko potrzebne kanały, żeby oszczędzić RAM)
    # Najpierw sprawdzamy metadane, żeby nie ładować całego pliku jeśli to niepotrzebne
    info = sf.info(wav_filename)
    n_channels = info.channels
    fs = info.samplerate
    total_samples = info.frames

    max_possible_order = int(np.sqrt(n_channels) - 1)
    if max_possible_order < 1 and n_channels < 4:
         raise ValueError(f"Not enough channels (min 4). Found: {n_channels}")

    if target_order is None or target_order > max_possible_order:
        order = max_possible_order
    else:
        order = target_order

    channels_used = (order + 1) ** 2
    print(f"Viz Processing Order: {order} (Using {channels_used} channels)")
    
    # Wczytujemy całe audio do RAM (dla typowych utworów to ok, dla >30min może być problem)
    # Czytamy tylko potrzebne kanały.
    # Używamy soundfile bezpośrednio dla szybkości, potem konwersja do float32
    audio_data, _ = sf.read(wav_filename, dtype='float32', always_2d=True)
    audio_data = audio_data[:, :channels_used]

    hop_size = int(fs / fps)
    num_frames = int(total_samples / hop_size)

    # 2. Przygotuj siatkę sferyczną
    sampling = pf.samplings.sph_equal_area(n_points)
    xyz_coords = sampling.cartesian
    sph_coords = np.column_stack((sampling.azimuth, sampling.elevation, sampling.radius))
    
    # Macierz transformacji (Points x Channels)
    Y_matrix = get_sh_matrix(sph_coords, order)

    energy_over_time = np.zeros((num_frames, n_points), dtype=np.float32)

    # 3. PRZETWARZANIE BLOKOWE (BATCH PROCESSING)
    # Zamiast pętli po 1 klatce, przetwarzamy np. 10 sekund (300 klatek) naraz.
    # To pozwala Numpy użyć szybkich operacji macierzowych BLAS.
    
    batch_frames = 200 # Liczba klatek wideo na jeden rzut
    samples_per_batch = batch_frames * hop_size
    
    total_batches = int(np.ceil(num_frames / batch_frames))

    for b in range(total_batches):
        # Raportowanie postępu
        if b % 5 == 0:
            percent = int((b / total_batches) * 100)
            progress_callback(percent)

        frame_start = b * batch_frames
        frame_end = min(frame_start + batch_frames, num_frames)
        
        sample_start = frame_start * hop_size
        sample_end = frame_end * hop_size
        
        # Pobieramy duży fragment audio
        chunk = audio_data[sample_start:sample_end, :] # (N_samples, Channels)
        
        # --- VECTORIZED MAGIC ---
        # 1. Rzutujemy wszystkie próbki naraz na sferę
        # (N_samples, Channels) dot (Channels, Points) -> (N_samples, Points)
        pressure_chunk = np.dot(chunk, Y_matrix.T)
        
        # 2. Musimy teraz uśrednić energię dla każdej klatki wideo.
        # Reshape: (Frames_in_batch, Samples_per_frame, Points)
        # Uwaga: Musimy przyciąć dane, jeśli ostatni batch jest niepełny
        valid_samples = (frame_end - frame_start) * hop_size
        pressure_chunk = pressure_chunk[:valid_samples]
        
        frames_in_this_batch = frame_end - frame_start
        
        # Zmieniamy kształt na 3D, żeby policzyć średnią wzdłuż osi czasu (axis 1)
        # shape: [liczba_klatek, probki_w_klatce, punkty_na_sferze]
        reshaped = pressure_chunk.reshape((frames_in_this_batch, hop_size, n_points))
        
        # RMS: Sqrt( Mean( Square( Signal ) ) )
        # axis=1 oznacza uśrednianie po próbkach wewnątrz jednej klatki
        energy_batch = np.sqrt(np.mean(reshaped ** 2, axis=1))
        
        # Zapisujemy wynik
        energy_over_time[frame_start:frame_end, :] = energy_batch

    progress_callback(100)
    
    # Bezpieczne obliczenie max_energy (unikanie dzielenia przez 0)
    max_energy = float(np.max(energy_over_time))
    if max_energy == 0: max_energy = 1.0
        
    return xyz_coords, energy_over_time, max_energy, fs, num_frames

class AudioPlayer(threading.Thread):
    def __init__(self, filename):
        super().__init__(daemon=True)
        self.filename = filename
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.start_time = 0
        self.pause_time = 0
        self.is_running = False
        self.gain = 1.0 
        self.current_yaw = -90.0
        self.current_pitch = 0.0

    def set_gain(self, vol_slider_val):
        self.gain = (vol_slider_val / 100.0) * 5.0

    def set_view_rotation(self, yaw, pitch):
        self.current_yaw = yaw
        self.current_pitch = pitch

    def run(self):
        data, fs = sf.read(self.filename, dtype='float32')
        n_channels = data.shape[1] if data.ndim > 1 else 1
        total_samples = len(data)

        if n_channels < 4:
            print("AudioPlayer: Not enough channels for rotation. Falling back to stereo/mono.")
        
        blocksize = 1024 
        idx = 0

        try:
            with sd.OutputStream(samplerate=fs, channels=2, blocksize=blocksize, dtype='float32') as stream:
                self.is_running = True
                self.start_time = time.time()

                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        sd.sleep(0.01) # Lepsze niż sd.sleep(10) dla responsywności
                        self.start_time = time.time() - self.pause_time
                        continue

                    self.pause_time = (idx / fs)
                    end = idx + blocksize

                    if end <= total_samples:
                        chunk = data[idx:end]
                        idx = end
                    else:
                        remainder = total_samples - idx
                        part1 = data[idx:]
                        part2 = data[:blocksize - remainder] 
                        chunk = np.concatenate((part1, part2))
                        idx = blocksize - remainder
                        self.start_time = time.time() - (idx / fs)
                    
                    # --- ROTATION LOGIC ---
                    if n_channels >= 4:
                        w = chunk[:, 0]
                        y = chunk[:, 1]
                        z = chunk[:, 2]
                        x = chunk[:, 3]

                        # Obliczamy kąt obrotu
                        theta = np.radians(-(self.current_yaw + 90))
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)

                        # Obracamy pole akustyczne (X i Y)
                        x_rot = x * cos_t - y * sin_t
                        y_rot = x * sin_t + y * cos_t
                        
                        # --- POPRAWIONE DEKODOWANIE STEREO ---
                        # Ustawiamy wirtualne mikrofony pod kątem +/- 45 stopni (Front-Left, Front-Right).
                        # Dzięki temu słyszymy to co z przodu (X) oraz to co z boków (Y).
                        # sqrt(2)/2 ~= 0.707
                        
                        sd_coef = 0.7071
                        
                        # Wzór na wirtualny mikrofon kardioidalny: 0.5 * W + 0.5 * Directivity
                        
                        # Lewy kanał (Mic skierowany na +45 stopni): Bierze +X i +Y
                        left = 0.5 * w + 0.5 * (sd_coef * x_rot + sd_coef * y_rot)
                        
                        # Prawy kanał (Mic skierowany na -45 stopni): Bierze +X i -Y
                        right = 0.5 * w + 0.5 * (sd_coef * x_rot - sd_coef * y_rot)
                        
                        out_stereo = np.column_stack((left, right))
                        out_stereo = out_stereo * self.gain
                    np.clip(out_stereo, -1.0, 1.0, out=out_stereo)
                    out_stereo = out_stereo.astype(np.float32)
                    
                    stream.write(out_stereo)

        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.is_running = False

    def get_current_time(self):
        if self._pause_event.is_set(): return self.pause_time
        if not self.is_running: return 0.0
        t = time.time() - self.start_time
        return max(0.0, t)

    def stop(self): self._stop_event.set()
    def pause(self): self._pause_event.set()
    def resume(self):
        self._pause_event.clear()
        self.start_time = time.time() - self.pause_time