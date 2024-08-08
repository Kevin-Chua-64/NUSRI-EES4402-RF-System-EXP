import numpy as np
import random
from scipy.special import erfc
import matplotlib.pyplot as plt

# Perform QPSK modulation 00->(1,1) 01->(-1,1) 11->(-1,-1) 10->(1,-1)
def qpsk_gray_modulation(bits, Eb):
    symbols = (1 - 2*bits[1::2]) + 1j * (1 - 2*bits[::2])
    symbols = symbols * np.sqrt(Eb)
    return symbols

# Simulate AWGN channel N(0,N_0/2)
def qpsk_add_awgn_noise(signal, noise_power):
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    received_signal = signal + noise
    return received_signal

# Perform QPSK demodulation D1={x>0,y>0}, D2={x<0,y>0}, D={x<0,y<0}, D4={x>0,y<0}
def qpsk_demodulation(received_symbols):
    demodulated_bits = np.zeros(2 * len(received_symbols), dtype=int)
    demodulated_bits[1::2] = np.real(received_symbols) < 0
    demodulated_bits[::2] = np.imag(received_symbols) < 0
    demodulated_bits[1::2][np.real(received_symbols)==0] = random.choices([0, 1], k=1)
    demodulated_bits[::2][np.imag(received_symbols)==0] = random.choices([0, 1], k=1)
    return demodulated_bits

# Calculate error bit number
def error_num(original_bits, received_bits):
    return np.sum(original_bits != received_bits)


# Simulation parameters
Eb = 1.0  # Fixed bit energy
num_bits = 10000  # Number of bits per simulation
error_up_num = 100  # At least error bits to find
qpsk_snr_range_dB = np.append(np.arange(-30, 10.4, 0.2), np.arange(10.4, 11.05, 0.05))  # SNR range in dB
qpsk_snr_range = 10**(qpsk_snr_range_dB/10)  # SNR range
qpsk_noise_range = Eb / qpsk_snr_range  # Noise range

# Initialize arrays to store BER values
ber_values_qpsk = np.zeros(len(qpsk_snr_range_dB))

# Perform simulation for each noise value
def QPSK():
    for i, noise in enumerate(qpsk_noise_range):
        error_count = 0
        loop_count = 0
        while(error_count <= error_up_num):  # Simulate until error bits is sufficient
            bits = np.random.randint(0, 2, num_bits)  # Generate random bits
            modulated_symbols = qpsk_gray_modulation(bits, Eb)
            received_symbols = qpsk_add_awgn_noise(modulated_symbols, noise)
            demodulated_bits = qpsk_demodulation(received_symbols)
            error_count += error_num(bits, demodulated_bits)
            loop_count += 1
            
        # Calculate BER
        ber_values_qpsk[i] = 1.0*error_count / (num_bits*loop_count)

# Simulation
QPSK()


# Q function
def Qx(x):
    return 0.5 * erfc(np.sqrt(0.5) * x)

# The theoretical curve
ber_theorem_qpsk = Qx(np.sqrt(2*qpsk_snr_range))
# The upper bound
upper_bound = 2*Qx(np.sqrt(2*qpsk_snr_range)) - Qx(np.sqrt(2*qpsk_snr_range))**2
# The lower bound
lower_bound = Qx(np.sqrt(2*qpsk_snr_range)) - 0.5*Qx(np.sqrt(2*qpsk_snr_range))**2


# Plot the curves
plt.figure(0, figsize=(10, 6))
plt.plot(qpsk_snr_range_dB, np.log10(ber_values_qpsk),  label='Simulation',  color='blue', marker='.')
plt.plot(qpsk_snr_range_dB, np.log10(ber_theorem_qpsk), label='Theoretical', color='red', linestyle='--')
qpsk_reach_10_6_snr = qpsk_snr_range_dB[np.argmin(np.abs(ber_values_qpsk - 1e-6))]  # SNR when BER reaches 10^-6
plt.text(qpsk_reach_10_6_snr-7, -6.25, r'BER:$10^{-6}$'+f'\nSNR:{qpsk_reach_10_6_snr:.2f}dB', fontsize=10)
plt.xlim([-33, 17])
plt.ylim([-7, 0])
plt.title('QPSK')
plt.xlabel('SNR(dB-scale)')
plt.ylabel('BER(log-scale)')
plt.legend()
plt.grid(True)

plt.figure(1, figsize=(10, 6))
plt.plot(qpsk_snr_range_dB, np.log10(ber_theorem_qpsk), label='Theoretical', color='red', linestyle='--')
plt.plot(qpsk_snr_range_dB, np.log10(upper_bound), label='Upper Bound', color='blue', linestyle='--')
plt.plot(qpsk_snr_range_dB, np.log10(lower_bound), label='Lower Bound', color='yellow', linestyle='--')
qpsk_upp_reach_10_6_snr = qpsk_snr_range_dB[np.argmin(np.abs(upper_bound - 1e-6))]  # SNR when BER reaches 10^-6
qpsk_low_reach_10_6_snr = qpsk_snr_range_dB[np.argmin(np.abs(lower_bound - 1e-6))]  # SNR when BER reaches 10^-6
plt.plot([qpsk_low_reach_10_6_snr,qpsk_upp_reach_10_6_snr], [-6,-6], color='green', linestyle='dashed', linewidth=3)
plt.text(qpsk_upp_reach_10_6_snr+0.5, -6.3, f'{qpsk_upp_reach_10_6_snr-qpsk_low_reach_10_6_snr:.2f}dB', color='green', fontsize=10)
plt.xlim([-33, 17])
plt.ylim([-7, 0])
plt.title('QPSK BER Bound')
plt.xlabel('SNR(dB-scale)')
plt.ylabel('BER(log-scale)')
plt.legend()
plt.grid(True)

plt.show()
