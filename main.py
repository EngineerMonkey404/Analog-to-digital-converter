import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq, fftshift
import math
import cmath

# harmonics = []
# for i in range(0, 4):
#     print(f'Введите {i + 1} гармонику')
#     ele = input()
#     harmonics.append(ele)
# print(harmonics)
N = 1000
t = 1.0 / 10000
A = np.array([2, 3, 8, 16])  # Массив амплитуд
F = np.array([3000.0, 2100, 3500, 1400])
W = F * 2 * np.pi
P = np.array([np.pi / 3, np.pi / 2, 0, np.pi / 4])
T = np.linspace(0.0, N * t, N, False)
U = A[0] * np.sin(T * W[0] + P[0]) + A[1] * np.sin(T * W[1] + P[1]) + A[2] * np.sin(T * W[2] + P[2]) + A[3] * np.sin(
    T * W[3] + P[3])
U_max = np.amax(U)
print("Максимальная амплитуда общего графика", U_max)
UF = fft(U)
fourier_amplitudes = 2.0 / N * np.abs(UF[0:N // 2])
harmonics_amplitudes = np.sort(fourier_amplitudes)[-4:]
harmonics_phases = []
for i in range(0, 4):
    j, = np.where(np.isclose(fourier_amplitudes, harmonics_amplitudes[i]))
    phase = np.angle(UF[j]) + np.pi / 2  # Относительно косинуса
    harmonics_phases.append(phase)
print("Фазы гармоник", harmonics_phases)
harmonics_freq = []
TF = fftfreq(1000, t)[:N // 2]
for i in range(0, 4):
    harmonics_freq.append(TF[list(fourier_amplitudes).index(harmonics_amplitudes[i])])
T_max = 1 / min(harmonics_freq)
harmonics_t = []
for i in range(0, 4):
    harmonics_t.append(1 / harmonics_freq[i])
print("Периоды синусоид", harmonics_t)
# np.lcm.reduce(harmonics_t)
# print(big_t)
time = np.linspace(0.0, T_max, N, False)
harmonics = []
print("Массив частот", harmonics_freq)
print("Массив амплитуд", harmonics_amplitudes)
for i in range(0, 4):
    harmonic = harmonics_amplitudes[i] * np.sin(
        time * 2 * math.pi * harmonics_freq[i] + harmonics_phases[i])
    harmonics.append(harmonic)
figure1, ax1 = plt.subplots(2, 2)
ax1[0][0].plot(T, U)
ax1[0][0].grid()
ax1[0][0].set_title('U(t)')
ax1[0][1].plot(TF, fourier_amplitudes)
ax1[0][1].set_title('U(f)')
ax1[0][1].grid()
ax1[1][0].plot(time, U)
figure2, ax2 = plt.subplots(4)
for i in range(0, 4):
    ax2[i].plot(time, harmonics[i])

discrete_freq = max(harmonics_freq) * 8
discrete_periode = 1 / discrete_freq
print("Периоды гармоник", harmonics_t)
print("Дискретная частота", discrete_freq, "Дискретный период", discrete_periode)
figure3, ax3 = plt.subplots(4)
# print(harmonics_t[2])
# print(harmonics_t[2] / discrete_periode)
harmonic_time = np.linspace(0.0, max(harmonics_t), round(max(harmonics_t) / discrete_periode))
first_harmonic_code = []
quants = np.linspace(0.0, U_max, 16)
print("Кванты", quants)
for i in range(0, 4):
    first_harmonic_code.append([])
    for step in harmonic_time:
        first_harmonic_code[i].append(
            harmonics_amplitudes[i] * np.sin(step * harmonics_freq[i] * 2 * math.pi + harmonics_phases[i]))
# for i in range(0, 4):
#     for amplitude in first_harmonic_code:


harmonic_code = []
for i in range(0, 4):
    harmonic_code.append([])
    for value in first_harmonic_code[i]:
        idx = (np.abs(quants - value)).argmin()
        print("Квант", quants[idx], "Значение", value)
        harmonic_code[i].append(idx)
print("Результат в каждый момент времени")
for j in range(len(harmonic_code[0])):
    sum = harmonic_code[0][j] + harmonic_code[1][j] + harmonic_code[2][j] + harmonic_code[3][j]
    print("Время", np.around(time[j], 8), harmonic_code[0][j], harmonic_code[1][j], harmonic_code[2][j], harmonic_code[3][j], sum, bin(sum))
for i in range(0, 4):
    ax3[i].plot(time, harmonics[i])
    ax3[i].step(harmonic_time, first_harmonic_code[i])
figure4, ax4 = plt.subplots(1)
ax4.plot(time, A[0] * np.sin(time * W[0] + P[0]))
plt.show()
