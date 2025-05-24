import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.signal import savgol_filter, wiener

# === Параметры ===
filename = r"music.wav"
output_image_before = "spectrogram_before.png"
output_image_after = "spectrogram_after.png"
filtered_wav_output = "guitar_filtered_all.wav"

# === Загрузка аудиофайла ===
rate, data = wav.read(filename)

# Переводим в моно, если стерео
if len(data.shape) > 1:
    print("Стерео: преобразуем в моно")
    data = data[:, 0]
print("Начинаем нормализацию")
# Нормализация
data = data.astype(np.float32)
data = data / np.max(np.abs(data))

# === Построение спектрограммы ===
def plot_spectrogram(signal_data, rate, title, filename):
    window = signal.windows.hann(1024)
    f, t, Sxx = signal.spectrogram(
        signal_data, fs=rate, window=window, nperseg=1024, noverlap=512,
        scaling='density', mode='magnitude'
    )
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
    plt.ylabel('Частота [Hz]')
    plt.xlabel('Время [s]')
    plt.title(title)
    plt.colorbar(label='Интенсивность [дБ]')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# === Спектрограмма до обработки ===
print("До обработки")
plot_spectrogram(data, rate, "Спектрограмма до фильтрации", output_image_before)
print("Начинаем обработку фильтрами")
# === Обработка фильтрами ===

# 1. Фильтр Савицкого-Голея
filtered = savgol_filter(data, window_length=101, polyorder=3)
print("Фильтр 1")
# 2. Фильтр Винера
filtered = wiener(filtered, mysize=101)
print("Фильтр 2")
# 3. Фильтр низких частот (ФНЧ)
cutoff_hz = 4000.0
nyquist = 0.5 * rate
norm_cutoff = cutoff_hz / nyquist
b, a = signal.butter(6, norm_cutoff, btype='low', analog=False)
filtered = signal.filtfilt(b, a, filtered)
print("Фильтр 3")
# === Сохраняем обработанный сигнал ===
filtered_int16 = np.int16(filtered / np.max(np.abs(filtered)) * 32767)
wav.write(filtered_wav_output, rate, filtered_int16)
print("Сохраненол")
# === Спектрограмма после фильтрации ===
plot_spectrogram(filtered, rate, "Спектрограмма после фильтрации (Савицкий + Винер + ФНЧ)", output_image_after)

print("✅ Готово:")
print(f"  • Спектрограмма ДО: {output_image_before}")
print(f"  • Спектрограмма ПОСЛЕ: {output_image_after}")
print(f"  • WAV файл после фильтрации: {filtered_wav_output}")
