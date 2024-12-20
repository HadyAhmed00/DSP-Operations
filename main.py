import cmath
import matplotlib.pyplot as plt
# 1. Adder
# Formula: output_signal[i] = sum(signal[i] for signal in signals)
# This function sums multiple signals element-wise.
def add_signals(signals):
    max_len = max(len(signal) for signal in signals)
    output_signal = [0] * max_len

    for signal in signals:
        for i in range(len(signal)):
            output_signal[i] += signal[i]

    return output_signal

# 2. Subtractor
# Formula: output_signal[i] = signal1[i] - signal2[i]
# This function subtracts the second signal from the first element-wise.
def subtract_signals(signal1, signal2):
    max_len = max(len(signal1), len(signal2))
    output_signal = [0] * max_len

    for i in range(max_len):
        val1 = signal1[i] if i < len(signal1) else 0
        val2 = signal2[i] if i < len(signal2) else 0
        output_signal[i] = val1 - val2

    return output_signal

# 3. Convolution
# Formula: output_signal[k] = sum(signal1[i] * signal2[k-i] for all valid i)
# This function computes the convolution of two signals.
def convolve_signals(signal1, signal2):
    n = len(signal1)
    m = len(signal2)
    output_signal = [0] * (n + m - 1)

    for i in range(n):
        for j in range(m):
            output_signal[i + j] += signal1[i] * signal2[j]

    return output_signal

# 4. FFT
# Formula: FFT[k] = sum(signal[n] * exp(-2j * pi * k * n / N) for n in range(N))
# This function recursively computes the Fast Fourier Transform.
def compute_fft(signal):
    n = len(signal)
    if n <= 1:
        return signal

    even = compute_fft(signal[0::2])
    odd = compute_fft(signal[1::2])

    t = [cmath.exp(-2j * cmath.pi * k / n) * odd[k % len(odd)] for k in range(n)]
    return [even[k % len(even)] + t[k] for k in range(n // 2)] + \
           [even[k % len(even)] - t[k] for k in range(n // 2)]

# 5. FIR Filter
# Formula: output_signal[k] = sum(signal[i] * taps[k-i] for all valid i)
# This function applies a simple FIR filter to a signal.
def apply_fir_filter(signal, num_taps, cutoff):
    taps = [cutoff if i < cutoff else 0 for i in range(num_taps)]
    return convolve_signals(signal, taps)

# 6. DST
# Formula: DST[k] = sum(signal[n] * sin(pi * (n+1) * (k+1) / (N+1)) for n in range(N))
# This function computes the Discrete Sine Transform (DST) of a signal.
def compute_dst(signal):
    n = len(signal)
    result = [0] * n
    for k in range(n):
        for i in range(n):
            result[k] += signal[i] * cmath.sin(cmath.pi * (i + 1) * (k + 1) / (n + 1)).real
    return result

# 7. Sampling
# Formula: output_signal = signal[::factor]
# This function downsamples a signal by the given factor.
def downsample_signal(signal, factor):
    return [signal[i] for i in range(0, len(signal), factor)]

# 8. Shifting
# Formula: output_signal[(i + shift_amount) % N] = signal[i]
# This function circularly shifts a signal by the given amount.
def shift_signal(signal, shift_amount):
    n = len(signal)
    output_signal = [0] * n

    for i in range(n):
        new_idx = (i + shift_amount) % n
        output_signal[new_idx] = signal[i]

    return output_signal

# 9. Quantization
# Formula: quantized_value = min_val + round((value - min_val) / step) * step
# This function quantizes a signal to a specified number of levels.
def quantize_signal(signal, num_levels):
    min_val = min(signal)
    max_val = max(signal)
    step = (max_val - min_val) / num_levels
    quantized_signal = []

    for value in signal:
        level = int((value - min_val) / step)
        quantized_signal.append(min_val + level * step)

    return quantized_signal
# Visualization
def visualize_signals_multiple(signals, labels, title):
    plt.figure(figsize=(12, 6))
    for signal, label in zip(signals, labels):
        plt.plot(signal, label=label)
    plt.title(title)
    plt.legend()
    plt.show()

# Example Usage
def main():
    t = [i * 0.1 for i in range(100)]
    signal1 = [cmath.sin(2 * cmath.pi * 0.05 * i).real for i in t]
    signal2 = [cmath.cos(2 * cmath.pi * 0.05 * i).real for i in t]

    added = add_signals([signal1, signal2])
    subtracted = subtract_signals(signal1, signal2)
    convolved = convolve_signals(signal1, signal2)
    fft_result = [abs(x) for x in compute_fft(signal1)]
    filtered = apply_fir_filter(signal1, num_taps=10, cutoff=5)
    dst_result = compute_dst(signal1)
    downsampled = downsample_signal(signal1, 2)
    shifted = shift_signal(signal1, 10)
    quantized = quantize_signal(signal1, 5)

    # Visualization for addition and subtraction
    visualize_signals_multiple([signal1, signal2, added], ["Signal 1", "Signal 2", "Added Signal"], "Addition of Signals")
    visualize_signals_multiple([signal1, signal2, subtracted], ["Signal 1", "Signal 2", "Subtracted Signal"], "Subtraction of Signals")

    # Visualization for other operations
    visualize_signals_multiple([signal1, convolved], ["Signal 1", "Convolved Signal"], "Convolution of Signals")
    visualize_signals_multiple([signal1, fft_result], ["Signal 1", "FFT Result"], "FFT of Signal")
    visualize_signals_multiple([signal1, filtered], ["Signal 1", "Filtered Signal"], "FIR Filter")
    visualize_signals_multiple([signal1, dst_result], ["Signal 1", "DST Result"], "DST of Signal")
    visualize_signals_multiple([signal1, downsampled], ["Signal 1", "Downsampled Signal"], "Downsampling")
    visualize_signals_multiple([signal1, shifted], ["Signal 1", "Shifted Signal"], "Signal Shifting")
    visualize_signals_multiple([signal1, quantized], ["Signal 1", "Quantized Signal"], "Quantization")

if __name__ == "__main__":
    main()