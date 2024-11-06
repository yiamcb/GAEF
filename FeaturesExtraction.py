def calculate_fractal_dimensions(data):
    # Calculate fractal dimensions using Higuchi Fractal Dimension (HFD)
    # You can replace this with another method if desired
    hfd_values = [dfa(data[channel]) for channel in range(data.shape[0])]
    return hfd_values

def calculate_lyapunov_exponents(data):
    # Calculate Lyapunov exponents
    lyap_values = [lyap_r(data[channel], emb_dim=10) for channel in range(data.shape[0])]
    return lyap_values

def calculate_spectral_entropy(data, fs=500):
    # Calculate spectral entropy
    entropy_values = []
    for channel in range(data.shape[0]):
        f, Pxx = welch(data[channel], fs, nperseg=256)
        spectral_entropy = entropy(Pxx, base=2)
        entropy_values.append(spectral_entropy)
    return entropy_values

def calculate_power_bands(data, fs=500):
    # Calculate power bands (alpha, beta, theta, gamma)
    power_bands = []
    for channel in range(data.shape[0]):
        f, Pxx = welch(data[channel], fs, nperseg=256)
        alpha_power = np.sum(Pxx[(f >= 8) & (f <= 13)])
        beta_power = np.sum(Pxx[(f > 13) & (f <= 30)])
        theta_power = np.sum(Pxx[(f >= 4) & (f <= 7)])
        gamma_power = np.sum(Pxx[(f > 30) & (f <= 50)])
        power_bands.append([alpha_power, beta_power, theta_power, gamma_power])
    return power_bands

def calculate_power_band_ratios(power_bands):
    # Calculate power band ratios
    ratios = []
    for band in power_bands:
        total_power = np.sum(band)
        ratios.append([x / total_power for x in band])
    return ratios


def extract_features(data, num_frames=50):
    num_samples = data.shape[1]
    num_channels = data.shape[0]

    # Calculate window size and overlap
    window_size = num_samples // num_frames
    overlap = window_size // 2
    print(window_size, flush=True)
    print(overlap, flush=True)

    Features = np.empty((18, 0))
    for i in range(0, num_samples - window_size + 1, window_size - overlap):
        frame = data[:, i:i+window_size]

        # Calculate features for each frame
        hfd_values = calculate_fractal_dimensions(frame)
        print(np.shape(hfd_values), flush=True)
        lyap_values = calculate_lyapunov_exponents(frame)
        print(np.shape(lyap_values), flush=True)
        entropy_values = calculate_spectral_entropy(frame)
        print(np.shape(entropy_values), flush=True)
        power_bands = calculate_power_bands(frame)
        print(np.shape(power_bands), flush=True)
        band_ratios = calculate_power_band_ratios(power_bands)
        print(np.shape(band_ratios), flush=True)

        combined_features = np.concatenate((np.array(hfd_values).reshape(-1, 1), np.array(lyap_values).reshape(-1, 1),
                                            np.array(entropy_values).reshape(-1, 1), np.array(power_bands),
                                            np.array(band_ratios)), axis=1)

        Features = np.concatenate((Features, combined_features), axis=1)

    return np.array(Features)