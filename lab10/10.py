import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# === Пути к файлам относительно 10.py ===
files = {
    "AAA": "AAA.wav",
    "III": "III.wav",
    "GAV": "GAV.wav"
}

# === Создание выходной папки ===
output_dir = "spectrograms"
os.makedirs(output_dir, exist_ok=True)

# === Функция спектрограммы (Задание 2) ===
def process_and_plot_spectrogram(filepath, label):
    rate, data = wav.read(filepath)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    data = data / np.max(np.abs(data))

    window = signal.windows.hann(1024)
    f, t, Sxx = signal.spectrogram(data, fs=rate, window=window, nperseg=1024, noverlap=512, scaling='density', mode='magnitude')
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
    plt.title(f"Спектрограмма сигнала: {label}")
    plt.xlabel("Время [с]")
    plt.ylabel("Частота [Гц]")
    plt.colorbar(label="Интенсивность [дБ]")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"spectrogram_{label}.png")
    plt.savefig(out_path)
    plt.close()

# === Задание 3: мин/макс частоты ===
def plot_min_max_freq(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))

    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=1/rate)
    power = np.abs(spectrum)

    threshold = np.max(power) * 0.01
    significant_freqs = freqs[power > threshold]
    min_f = np.min(significant_freqs)
    max_f = np.max(significant_freqs)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power)
    plt.axvline(min_f, color='green', linestyle='--', label=f"Мин: {min_f:.1f} Гц")
    plt.axvline(max_f, color='red', linestyle='--', label=f"Макс: {max_f:.1f} Гц")
    plt.title(f"Задание 3: Частоты сигнала {label}")
    plt.xlabel("Частота [Гц]")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"3_{label}_min_max_freq.png"))
    plt.close()

# === Задание 4: основной тон и обертоны ===
def plot_harmonics(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))

    window_size = int(rate * 0.05)
    window = signal.windows.hann(window_size)
    segment = data[:window_size] * window
    spectrum = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(window_size, d=1/rate)
    power = np.abs(spectrum) ** 2

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power)
    plt.title(f"Задание 4: Обертоны сигнала {label}")
    plt.xlabel("Частота [Гц]")
    plt.ylabel("Энергия")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"4_{label}_harmonics.png"))
    plt.close()

# === Задание 5: три сильных форманты ===
def plot_formants(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))

    window_size = int(rate * 0.1)
    step = window_size
    freqs = np.fft.rfftfreq(window_size, d=1 / rate)

    max_energies = []
    time_stamps = []

    for start in range(0, len(data) - window_size, step):
        segment = data[start:start + window_size]
        spectrum = np.fft.rfft(segment * signal.windows.hann(window_size))
        power = np.abs(spectrum) ** 2
        max_idx = np.argsort(power)[-3:]
        max_energies.append(freqs[max_idx])
        time_stamps.append(start / rate)

    max_energies = np.array(max_energies)

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(time_stamps, max_energies[:, i], label=f"Форманта {i+1}")
    plt.title(f"Задание 5: Форманты сигнала {label}")
    plt.xlabel("Время [с]")
    plt.ylabel("Частота [Гц]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"5_{label}_formants.png"))
    plt.close()

# === Запуск всех заданий ===
for label, filepath in files.items():
    process_and_plot_spectrogram(filepath, label)
    plot_min_max_freq(filepath, label)
    plot_harmonics(filepath, label)
    plot_formants(filepath, label)

print("✅ Все задания выполнены и сохранены в папку:", output_dir)
