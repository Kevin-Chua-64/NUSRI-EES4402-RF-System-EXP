import numpy as np
import random
from scipy.special import erfc
import matplotlib.pyplot as plt

# Perform 2-PAM modulation 0->(-1) 1->(1)
def pam2_gray_modulation(bits, Eb):
    symbols = -1 + 2*bits
    symbols = symbols * np.sqrt(Eb)
    return symbols

# Perform 4-PAM modulation 00->(-3) 01->(-1) 11->(1) 10->(3)
def pam4_gray_modulation(bits, Eb):
    symbols = (-1 + 2*bits[::2]) *  (3 - 2*bits[1::2])
    symbols = symbols * np.sqrt(0.4*Eb)
    return symbols

# Simulate AWGN channel N(0,N_0/2)
def pam_add_awgn_noise(signal, noise_power):
    noise = np.sqrt(noise_power/2) * np.random.randn(*signal.shape)
    received_signal = signal + noise
    return received_signal

# Perform 2-PAM demodulation D1={r<0}, D2={r>0}
def pam2_demodulation(received_symbols):
    demodulated_bits = np.zeros(len(received_symbols), dtype=int)
    demodulated_bits = received_symbols > 0
    demodulated_bits[received_symbols==0] = random.choices([0, 1], k=1)
    return demodulated_bits

# Perform 4-PAM demodulation D1={r<-2sqrt(0.4*Eb)}, D2={-2sqrt(0.4*Eb)<r<0}, D3={0<r<2sqrt(0.4*Eb)}, D4={r>2sqrt(0.4*Eb)}
def pam4_demodulation(received_symbols):
    demodulated_bits = np.zeros(2 * len(received_symbols), dtype=int)
    demodulated_bits[::2] = received_symbols > 0
    demodulated_bits[::2][received_symbols==0] = random.choices([0, 1], k=1)
    demodulated_bits[1::2] = (received_symbols > -2*np.sqrt(0.4*Eb)) & (received_symbols < 2*np.sqrt(0.4*Eb))
    demodulated_bits[1::2][(received_symbols == -2*np.sqrt(0.4*Eb)) | (received_symbols == 2*np.sqrt(0.4*Eb))] = random.choices([0, 1], k=1)
    return demodulated_bits

# Calculate error bit number
def error_num(original_bits, received_bits):
    return np.sum(original_bits != received_bits)


# Simulation parameters
Eb = 1.0  # Fixed bit energy
num_bits = 10000  # Number of bits per simulation
error_up_num = 100  # At least error bits to find
# SNR range in dB
pam2_snr_range_dB = np.append(np.arange(-30, 10.4, 0.2), np.arange(10.4, 11.05, 0.05))
pam4_snr_range_dB = np.append(np.arange(-30, 14.2, 0.2), np.arange(14.2, 15.05, 0.05))
# SNR range
pam2_snr_range = 10**(pam2_snr_range_dB/10)
pam4_snr_range = 10**(pam4_snr_range_dB/10)
# Noise range
pam2_noise_range = Eb / pam2_snr_range
pam4_noise_range = Eb / pam4_snr_range

# Initialize arrays to store BER values
ber_values_2pam = np.zeros(len(pam2_snr_range_dB))
ber_values_4pam = np.zeros(len(pam4_snr_range_dB))

# Perform simulation for each noise value (2-PAM)
def PAM2():
    for i, noise in enumerate(pam2_noise_range):
        error_count = 0
        loop_count = 0
        while(error_count <= error_up_num):  # Simulate until error bits is sufficient
            bits = np.random.randint(0, 2, num_bits)  # Generate random bits
            modulated_symbols = pam2_gray_modulation(bits, Eb)
            received_symbols = pam_add_awgn_noise(modulated_symbols, noise)
            demodulated_bits = pam2_demodulation(received_symbols)
            error_count += error_num(bits, demodulated_bits)
            loop_count += 1
            
        # Calculate BER
        ber_values_2pam[i] = 1.0*error_count / (num_bits*loop_count)


# Perform simulation for each noise value (4-PAM)
def PAM4():
    for i, noise in enumerate(pam4_noise_range):
        error_count = 0
        loop_count = 0
        while(error_count <= error_up_num):  # Simulate until error bits is sufficient
            bits = np.random.randint(0, 2, num_bits)  # Generate random bits
            modulated_symbols = pam4_gray_modulation(bits, Eb)
            received_symbols = pam_add_awgn_noise(modulated_symbols, noise)
            demodulated_bits = pam4_demodulation(received_symbols)
            error_count += error_num(bits, demodulated_bits)
            loop_count += 1
            
        # Calculate BER
        ber_values_4pam[i] = 1.0*error_count / (num_bits*loop_count)

# Simulation
PAM2()
PAM4()


# Q function
def Qx(x):
    return 0.5 * erfc(np.sqrt(0.5) * x)

# The theoretical curve
ber_theorem_2pam = Qx(np.sqrt(2*pam2_snr_range))
ber_theorem_4pam = 0.75*Qx(np.sqrt(0.8*pam4_snr_range)) + 0.5*Qx(np.sqrt(7.2*pam4_snr_range)) - 0.25*Qx(np.sqrt(20*pam4_snr_range))


# Plot the curves
plt.figure(0, figsize=(10, 6))
plt.plot(pam2_snr_range_dB, np.log10(ber_values_2pam),  label='Simulation',  color='blue', marker='.')
plt.plot(pam2_snr_range_dB, np.log10(ber_theorem_2pam), label='Theoretical', color='red', linestyle='--')
pam2_reach_10_6_snr = pam2_snr_range_dB[np.argmin(np.abs(ber_values_2pam - 1e-6))]  # SNR when BER reaches 10^-6
plt.text(pam2_reach_10_6_snr-7, -6.25, r'BER:$10^{-6}$'+f'\nSNR:{pam2_reach_10_6_snr:.2f}dB', fontsize=10)
plt.xlim([-33, 17])
plt.ylim([-7, 0])
plt.title('2-PAM')
plt.xlabel('SNR(dB-scale)')
plt.ylabel('BER(log-scale)')
plt.legend()
plt.grid(True)

plt.figure(1, figsize=(10, 6))
plt.plot(pam4_snr_range_dB, np.log10(ber_values_4pam),  label='Simulation',  color='blue', marker='.')
plt.plot(pam4_snr_range_dB, np.log10(ber_theorem_4pam), label='Theoretical', color='red', linestyle='--')
pam4_reach_10_6_snr = pam4_snr_range_dB[np.argmin(np.abs(ber_values_4pam - 1e-6))]  # SNR when BER reaches 10^-6
plt.text(pam4_reach_10_6_snr-7, -6.25, r'BER:$10^{-6}$'+f'\nSNR:{pam4_reach_10_6_snr:.2f}dB', fontsize=10)
plt.xlim([-33, 17])
plt.ylim([-7, 0])
plt.title('4-PAM')
plt.xlabel('SNR(dB-scale)')
plt.ylabel('BER(log-scale)')
plt.legend()
plt.grid(True)

plt.figure(2, figsize=(10, 6))
plt.plot(pam2_snr_range_dB, np.log10(ber_theorem_2pam), label='2-PAM', color='blue', linestyle='--')
plt.plot(pam4_snr_range_dB, np.log10(ber_theorem_4pam), label='4-PAM', color='red', linestyle='--')
plt.plot([pam2_reach_10_6_snr,pam4_reach_10_6_snr], [-6,-6], color='green', linestyle='dashed', linewidth=3)
plt.text(pam2_reach_10_6_snr+0.5, -6.3, f'{pam4_reach_10_6_snr-pam2_reach_10_6_snr:.2f}dB', color='green', fontsize=10)
plt.xlim([-33, 17])
plt.ylim([-7, 0])
plt.title('2-PAM and 4-PAM Comparison (Theoretical)')
plt.xlabel('SNR(dB-scale)')
plt.ylabel('BER(log-scale)')
plt.legend()
plt.grid(True)

plt.show()
