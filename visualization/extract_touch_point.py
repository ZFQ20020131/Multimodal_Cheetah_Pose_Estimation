# -*- coding: utf-8 -*-
"""
Contact-event detection from audio using STFT band energy (50-300 Hz),
with robust threshold, hysteresis, and min-duration. Then align to 36 frames
and produce Fig-B and Fig-C.
/
Outputs:
  /mnt/data/FigB_BandEnergy_Windows.png
  /mnt/data/FigC_BinaryMask_Frames.png
  /mnt/data/contact_events.json   # events in seconds and frames
"""
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Config -------------------------
AUDIO_PATH   = "./data/fauna/audio_segment.wav"  # your audio
BAND_LO_HZ   = 50
BAND_HI_HZ   = 300

# Robust thresholding
MAD_K        = 6.0          # threshold = median + k * MAD
HYST_RATIO   = 0.8          # low = HYST_RATIO * high (for hysteresis)

# Event rules
MIN_DURATION = 0.06         # seconds, min continuous duration above low
MIN_GAP      = 0.03         # seconds, merge events closer than this

# Frame mapping (evenly split the whole clip)
TOTAL_FRAMES = 36           # frame indices will be [0..35]

# Output paths
OUT_FIG_B    = "/mnt/data/FigB_BandEnergy_Windows.png"
OUT_FIG_C    = "/mnt/data/FigC_BinaryMask_Frames.png"
OUT_JSON     = "/mnt/data/contact_events.json"

# ----------------------------------------------------------

def load_audio(path):
    """Return (y, sr) normalized to float32 in [-1,1]."""
    try:
        import soundfile as sf
        y, sr = sf.read(path, always_2d=False)
    except Exception:
        # Fallback: librosa
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    y = np.asarray(y, dtype=np.float32)
    return y, sr

def stft_band_energy(y, sr, band_lo=BAND_LO_HZ, band_hi=BAND_HI_HZ,
                     win_ms=25, hop_ms=10, use_librosa=True):
    """
    Compute STFT power spectrogram, then sum power in [band_lo, band_hi] per frame.
    Returns:
        band_energy: (T,) power-like values
        t_sec:       (T,) center time per STFT frame in seconds
    """
    n_fft = int(sr * win_ms / 1000.0)
    n_fft = int(2 ** math.ceil(np.log2(max(256, n_fft))))  # power of 2, >=256
    hop   = int(sr * hop_ms / 1000.0)
    win   = n_fft

    try:
        if use_librosa:
            import librosa
            S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True)
            mag = np.abs(S) ** 2  # power
            freqs = np.linspace(0, sr/2, 1 + n_fft//2)
            # librosa returns full FFT (n_fft rows); keep only positive freqs
            mag = mag[:1 + n_fft//2, :]
        else:
            raise ImportError
    except Exception:
        from scipy.signal import stft
        f, tt, Z = stft(y, fs=sr, nperseg=win, noverlap=win-hop, nfft=n_fft, window="hann", padded=True)
        mag  = np.abs(Z) ** 2
        freqs = f

    # band bins
    lo_bin = np.searchsorted(freqs, band_lo, side="left")
    hi_bin = np.searchsorted(freqs, band_hi, side="right")
    lo_bin = np.clip(lo_bin, 0, mag.shape[0]-1)
    hi_bin = np.clip(hi_bin, lo_bin+1, mag.shape[0])

    band_energy = mag[lo_bin:hi_bin, :].sum(axis=0)

    # time axis (center of each frame)
    T = band_energy.shape[0]
    t_sec = (np.arange(T) * hop + win/2) / float(sr)
    return band_energy.astype(np.float64), t_sec

def robust_threshold(x, k=MAD_K):
    """
    median + k * MAD (median absolute deviation).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    thr = med + k * mad
    return thr

def hysteresis_segments(t, x, high_thr, low_thr, min_dur=MIN_DURATION, min_gap=MIN_GAP):
    """
    Detect segments using hysteresis:
      - rising when x crosses high_thr
      - falling when x goes below low_thr
      - keep segments of duration >= min_dur
      - merge neighboring segments with gaps < min_gap
    Returns list of (t_start, t_end).
    """
    events = []
    in_seg = False
    seg_start = None
    for i in range(len(x)):
        if not in_seg and x[i] >= high_thr:
            in_seg = True
            seg_start = t[i]
        elif in_seg and x[i] <= low_thr:
            in_seg = False
            seg_end = t[i]
            if seg_end - seg_start >= min_dur:
                events.append([seg_start, seg_end])
    # Tail
    if in_seg:
        seg_end = t[-1]
        if seg_end - seg_start >= min_dur:
            events.append([seg_start, seg_end])

    # Merge close segments
    if not events:
        return []
    merged = [events[0]]
    for s, e in events[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged

def map_events_to_frames(events, total_frames, clip_duration):
    """
    Evenly split [0, clip_duration] into total_frames bins; mark frames that overlap any event.
    Returns:
        m_frame: (total_frames,) int {0,1}
        frame_edges: (total_frames+1,) edges in seconds
        frame_events: list of (start_frame, end_frame) inclusive
    """
    edges = np.linspace(0.0, clip_duration, total_frames + 1)
    m = np.zeros(total_frames, dtype=int)
    frame_events = []

    for (ts, te) in events:
        # find frames overlapped by [ts, te]
        start_f = int(np.searchsorted(edges, ts, side="right") - 1)
        end_f   = int(np.searchsorted(edges, te, side="left"))
        start_f = np.clip(start_f, 0, total_frames-1)
        end_f   = np.clip(end_f,   start_f, total_frames-1)
        m[start_f:end_f+1] = 1
        frame_events.append((int(start_f), int(end_f)))
    return m, edges, frame_events

def annotate_time(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="bottom",
            fontsize=12, color="#11698e",
            bbox=dict(boxstyle="round,pad=0.25", fc="#e6f7ff", ec="#11698e", lw=1.5))

def plot_figB(t, band_energy, thr, events, out_path):
    plt.figure(figsize=(14, 4.5))
    plt.plot(t, band_energy, color="#f5a536", lw=3, label=f"Band energy ({BAND_LO_HZ}-{BAND_HI_HZ} Hz)")
    plt.axhline(thr, color="red", ls="--", lw=2, label="Threshold")

    for (ts, te) in events:
        plt.axvspan(ts, te, color="#a3f3ff", alpha=0.35, label="Event window" if "Event window" not in plt.gca().get_legend_handles_labels()[1] else None)
        # annotate start/end times near threshold line
        y = thr + 0.02 * (band_energy.max() - band_energy.min())
        annotate_time(plt.gca(), ts, y, f"{ts:.3f}s")
        annotate_time(plt.gca(), te, y, f"{te:.3f}s")

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Energy (a.u.)", fontsize=14)
    plt.title("Fig-B  Band energy with frame-aligned event windows", fontsize=16)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_figC(m_frame, frame_events, out_path):
    N = len(m_frame)
    x = np.arange(N)

    plt.figure(figsize=(14, 3.8))
    # step plot for binary mask
    plt.step(x, m_frame, where="post", color="#cc8400", lw=3, label=r"Contact mask $m_t$")
    # shade event windows in frame domain
    for (fs, fe) in frame_events:
        plt.axvspan(fs, fe+1-1e-9, color="#a3f3ff", alpha=0.35, label="Event window" if "Event window" not in plt.gca().get_legend_handles_labels()[1] else None)
        # annotate
        plt.text(fs, 1.05, f"start={fs}", ha="center", va="bottom",
                 fontsize=11, color="#11698e",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#e6f7ff", ec="#11698e", lw=1.5))
        plt.text(fe, 1.05, f"end={fe}",   ha="center", va="bottom",
                 fontsize=11, color="#11698e",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#e6f7ff", ec="#11698e", lw=1.5))

    plt.ylim(-0.1, 1.2)
    plt.xlim(0, N-1)
    plt.xlabel("Frame index", fontsize=14)
    plt.ylabel(r"$m_t$", fontsize=16)
    plt.title("Fig-C  Binary contact mask per frame", fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main():
    # 1) Load audio
    y, sr = load_audio(AUDIO_PATH)
    duration = len(y) / float(sr)

    # 2) STFT band energy
    band_energy, t_sec = stft_band_energy(y, sr)

    # 3) Robust threshold + hysteresis detection
    thr_high = robust_threshold(band_energy, k=MAD_K)
    thr_low  = HYST_RATIO * thr_high
    events   = hysteresis_segments(t_sec, band_energy, thr_high, thr_low,
                                   min_dur=MIN_DURATION, min_gap=MIN_GAP)

    # 4) Map events to 36 frames
    m_frame, frame_edges, frame_events = map_events_to_frames(events, TOTAL_FRAMES, duration)

    # 5) Save JSON (for reproducibility)
    out = {
        "sr": int(sr),
        "duration_sec": float(duration),
        "band_energy_len": int(len(band_energy)),
        "threshold_high": float(thr_high),
        "threshold_low": float(thr_low),
        "events_sec": [[float(s), float(e)] for (s, e) in events],
        "total_frames": int(TOTAL_FRAMES),
        "frame_events": [[int(s), int(e)] for (s, e) in frame_events],
        "mask_frames": [int(v) for v in m_frame.tolist()],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # 6) Plots
    plot_figB(t_sec, band_energy, thr_high, events, OUT_FIG_B)
    plot_figC(m_frame, frame_events, OUT_FIG_C)

    print(f"Done.\nSaved:\n  {OUT_FIG_B}\n  {OUT_FIG_C}\n  {OUT_JSON}")

if __name__ == "__main__":
    main()







