import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.spatial.transform as st
# import matplotlib.pyplot as plt


def add_gaussian_noise(seq, σ=0.01):
    """
    Add zero-mean Gaussian noise. Does not change
    actual stride length or step width.
    """
    noise = np.random.normal(0, σ, size=seq.shape)
    return seq + noise


def smooth_signal(seq, σ=0.5):
    """
    Apply Gaussian smoothing along time. Does not
    alter the true gait geometry, only attenuates
    high-frequency jitter.
    """
    return np.stack([gaussian_filter1d(seq[:, i], σ) for i in range(seq.shape[1])], axis=1)


def random_bias_offset(seq, max_bias=0.05):
    """
    Add a constant 3-axis bias to each triad. Simulates
    a small calibration drift. Does not change true
    stride length if the network learns to ignore
    constant offsets correctly.
    """
    T, C = seq.shape
    if C % 3 != 0:
        return seq
    biased = seq.copy()
    for i in range(0, C, 3):
        bias = np.random.uniform(-max_bias, max_bias, size=(1, 3))
        biased[:, i : i + 3] += bias
    return biased


def random_rotation(seq, max_angle_deg=10):
    """
    Apply one small rigid rotation to each 3-axis block.
    Because rotation does NOT change actual stride length,
    the true label remains valid.
    """
    T, C = seq.shape
    if C % 3 != 0:
        return seq

    angles = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
    rot = st.Rotation.from_euler('xyz', angles)
    rotated = np.zeros_like(seq)
    for i in range(0, C, 3):
        block = seq[:, i : i + 3]       # (T, 3)
        rotated_block = rot.apply(block)  # (T, 3)
        rotated[:, i : i + 3] = rotated_block
    return rotated


def freq_mask(seq, num_masks=1, max_band=3):
    """
    Zero out a few random frequency bins. This emulates
    narrowband interference. The geometric stride label is
    unaffected, so it's still valid.
    """
    T, C = seq.shape
    fft_seq = np.fft.rfft(seq, axis=0)  # (T//2+1, C)
    n_bins = fft_seq.shape[0]
    for _ in range(num_masks):
        f0 = np.random.randint(0, n_bins - max_band)
        fft_seq[f0 : f0 + max_band, :] = 0
    return np.fft.irfft(fft_seq, n=T, axis=0)


def channel_dropout(seq, drop_prob=0.1, max_len=10):
    """
    Zero‐mask one 3-axis block for a short window, simulating
    sensor dropout. Still does not change the true spatial
    relationship—just hides it briefly.
    """
    T, C = seq.shape
    if C % 3 != 0:
        return seq
    if np.random.rand() < drop_prob:
        block_idx = np.random.randint(0, C // 3)
        start_frame = np.random.randint(0, T - max_len + 1)
        drop_len = np.random.randint(1, max_len + 1)
        seq[start_frame : start_frame + drop_len, block_idx * 3 : block_idx * 3 + 3] = 0.0
    return seq


def augment_sequence(
    seq,
    visualize=False,
    p_noise=0.5,
    p_rot=0.3,
    p_smooth=0.2,
    p_freq_mask=0.2,
    p_bias=0.3,
    p_chan_drop=0.2
):
    """
    Apply only “label-safe” augmentations:
     - Gaussian noise
     - Small 3D rotation
     - Gaussian smoothing
     - Frequency masking
     - Small bias offset (calibration drift)
     - Brief channel dropout

    None of these change the true stride length or width label.
    """
    seq = seq.copy()
    original_seq = seq.copy()
    applied = []

    if np.random.rand() < p_noise:
        seq = add_gaussian_noise(seq, σ=0.01)
        applied.append("GaussianNoise")

    if np.random.rand() < p_bias:
        seq = random_bias_offset(seq, max_bias=0.03)
        applied.append("BiasOffset")

    if np.random.rand() < p_rot:
        seq = random_rotation(seq, max_angle_deg=10)
        applied.append("Rotation")

    if np.random.rand() < p_smooth:
        seq = smooth_signal(seq, σ=0.5)
        applied.append("GaussianSmooth")

    if np.random.rand() < p_freq_mask:
        seq = freq_mask(seq, num_masks=1, max_band=3)
        applied.append("FreqMask")

    if np.random.rand() < p_chan_drop:
        seq = channel_dropout(seq, drop_prob=1.0, max_len=10)
        applied.append("ChannelDrop")

    # if visualize:
    #     T, C = original_seq.shape
    #     fig, axs = plt.subplots(C, 1, figsize=(8, 2 * C))
    #     for i in range(C):
    #         axs[i].plot(original_seq[:, i], label='Original', alpha=0.7)
    #         axs[i].plot(seq[:, i], label='Augmented', alpha=0.7)
    #         axs[i].set_ylabel(f'Ch {i+1}')
    #         if i == 0:
    #             axs[i].legend()
    #     axs[-1].set_xlabel('Time Index')
    #
    #     title_txt = "Applied: " + (", ".join(applied) if applied else "None")
    #     fig.suptitle(title_txt, fontsize=12, y=0.95)
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #     plt.show()

    return seq
