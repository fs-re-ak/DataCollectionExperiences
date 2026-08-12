"""
generate_report.py — Produce a self-contained HTML validation report for one
or more recorded sessions found under DATA_DIR.

Usage
─────
    python generate_report.py                  # processes all sessions in data/
    python generate_report.py --session HERMES_F001_S01_20260811

Pipeline per session
────────────────────
1. Load EXG (timestamp + ch1…ch8) and MARKERS CSVs.
2. Butterworth bandpass 15–35 Hz (order 4, zero-phase sosfiltfilt).
3. Rolling RMS per channel, window = 2 s × SAMPLE_RATE samples.
4. Per-sample features:
       unit_vec  = rms_8d / ‖rms_8d‖   (8-dim, relative channel contribution)
       rms_amp   = ‖rms_8d‖             (scalar, overall activation)
5. Parse HOLD_… / RELAX_… marker pairs → hold windows per emotion per trial.
6. Slice features to hold windows; aggregate mean ± std across trials/blocks.
7. One figure per emotion (two subplots: unit-vec components, RMS amplitude).
8. Embed all figures as base64 PNG in a single self-contained HTML file written
   next to the source data.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger("generate_report")


# ── Constants ──────────────────────────────────────────────────────────────────

FS            = config.SAMPLE_RATE          # 250 Hz
N_CH          = config.N_CHANNELS           # 8
CH_LABELS     = config.CHANNEL_LABELS       # list[str], len 8
DATA_DIR      = Path(config.DATA_DIR)

BP_LOW        = 15.0    # Hz
BP_HIGH       = 35.0    # Hz
FILTER_ORDER  = 4
RMS_WIN_SEC   = 2.0     # seconds
RMS_WIN_SAMP  = int(RMS_WIN_SEC * FS)       # 500 samples

EMOTION_COLORS = {
    "HAPPINESS":      "#f5c542",
    "ANGER":          "#e05252",
    "DISGUST":        "#7db87d",
    "FEAR":           "#9b59b6",
    "SURPRISE":       "#3498db",
    "SADNESS":        "#5b8ac9",
    "NEUTRAL":        "#95a5a6",
    "CONTEMPT_LEFT":  "#e67e22",
    "CONTEMPT_RIGHT": "#d35400",
}


# ── Signal processing helpers ──────────────────────────────────────────────────

def bandpass(data: np.ndarray, fs: float = FS,
             low: float = BP_LOW, high: float = BP_HIGH,
             order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied column-wise."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=0)


def rolling_rms(data: np.ndarray, window: int = RMS_WIN_SAMP) -> np.ndarray:
    """
    Compute a causal rolling RMS along axis 0 using a cumulative-sum trick.
    Returns an array of the same shape; the first (window-1) rows are NaN.
    """
    out = np.full_like(data, np.nan, dtype=float)
    cs = np.cumsum(data ** 2, axis=0)
    out[window - 1] = np.sqrt(cs[window - 1] / window)
    out[window:] = np.sqrt((cs[window:] - cs[:-window]) / window)
    return out


# ── Marker parsing ─────────────────────────────────────────────────────────────

HOLD_RE  = re.compile(r"^HOLD_([A-Z_]+)_B(\d+)_T(\d+)$")
RELAX_RE = re.compile(r"^RELAX_([A-Z_]+)_B(\d+)_T(\d+)$")


def parse_hold_windows(markers: pd.DataFrame) -> list[dict]:
    """
    Return a list of dicts:
        {"emotion": str, "block": int, "trial": int,
         "t_start": float, "t_end": float}
    where t_start/t_end are Unix timestamps bracketing the HOLD period.
    """
    hold_map: dict[tuple, float] = {}
    windows = []
    for _, row in markers.iterrows():
        mstr = str(row["marker"]).strip()
        m = HOLD_RE.match(mstr)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            hold_map[key] = float(row["timestamp"])
            continue
        m = RELAX_RE.match(mstr)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            if key in hold_map:
                windows.append({
                    "emotion": key[0],
                    "block":   key[1],
                    "trial":   key[2],
                    "t_start": hold_map.pop(key),
                    "t_end":   float(row["timestamp"]),
                })
    return windows


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(exg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given EXG DataFrame with columns [timestamp, ch1…ch8]:
      1. Bandpass filter channels.
      2. Compute rolling RMS.
      3. Compute unit vector and RMS amplitude per sample.

    Returns
    -------
    timestamps : (N,)        Unix timestamps (float64)
    unit_vec   : (N, 8)      Per-sample unit vector of RMS
    rms_amp    : (N,)        Per-sample ‖rms_8d‖
    """
    ts = exg["timestamp"].to_numpy(dtype=float)
    raw = exg[[f"ch{i}" for i in range(1, N_CH + 1)]].to_numpy(dtype=float)

    filtered = bandpass(raw)
    rms = rolling_rms(filtered)

    norm = np.linalg.norm(rms, axis=1, keepdims=True)          # (N, 1)
    safe_norm = np.where(norm == 0, 1.0, norm)                  # avoid div-by-zero
    unit_vec = rms / safe_norm                                   # (N, 8)
    rms_amp  = norm.squeeze(1)                                   # (N,)

    return ts, unit_vec, rms_amp


# ── Per-emotion statistics ─────────────────────────────────────────────────────

def compute_emotion_stats(
    timestamps: np.ndarray,
    unit_vec:   np.ndarray,
    rms_amp:    np.ndarray,
    windows:    list[dict],
) -> dict[str, dict]:
    """
    For each emotion, collect all hold-window samples across blocks/trials,
    then compute mean and std per channel for unit_vec and rms_amp.

    Returns
    -------
    {
      emotion_key: {
        "unit_mean": np.ndarray (8,),
        "unit_std":  np.ndarray (8,),
        "amp_mean":  np.ndarray (8,),   # mean rms_amp repeated per channel
        "amp_std":   np.ndarray (8,),   # std  rms_amp repeated per channel
        "n_samples": int,
        "n_trials":  int,
      }
    }
    """
    # Group windows by emotion
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for w in windows:
        grouped[w["emotion"]].append(w)

    stats: dict[str, dict] = {}
    for emotion, wins in grouped.items():
        uv_all  = []
        amp_all = []
        for w in wins:
            mask = (timestamps >= w["t_start"]) & (timestamps < w["t_end"])
            uv_slice  = unit_vec[mask]
            amp_slice = rms_amp[mask]
            # Drop NaN rows (from rolling-RMS warm-up)
            valid = ~np.any(np.isnan(uv_slice), axis=1) & ~np.isnan(amp_slice)
            uv_all.append(uv_slice[valid])
            amp_all.append(amp_slice[valid])

        uv_cat  = np.concatenate(uv_all,  axis=0) if uv_all  else np.empty((0, N_CH))
        amp_cat = np.concatenate(amp_all, axis=0) if amp_all else np.empty((0,))

        n = len(uv_cat)
        stats[emotion] = {
            "unit_mean": uv_cat.mean(axis=0)  if n else np.zeros(N_CH),
            "unit_std":  uv_cat.std(axis=0)   if n else np.zeros(N_CH),
            "amp_mean":  amp_cat.mean()        if n else 0.0,
            "amp_std":   amp_cat.std()         if n else 0.0,
            "n_samples": n,
            "n_trials":  len(wins),
        }
    return stats


# ── Plotting ───────────────────────────────────────────────────────────────────

SHORT_CH = [
    "Supra_L", "Supra_R", "Zygo_L", "Zygo_R",
    "Temp_EEG", "Glabella", "Temp_R", "Nasolab",
]

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_emotion(emotion: str, stat: dict, color: str) -> str:
    """
    Two-subplot figure:
      top    – mean ± std of unit-vector components per channel
      bottom – mean ± std of RMS amplitude per channel
    Returns base64-encoded PNG string.
    """
    x = np.arange(N_CH)
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#ccccdd", labelsize=8)
        ax.spines[:].set_color("#44445a")
        ax.yaxis.label.set_color("#ccccdd")
        ax.xaxis.label.set_color("#ccccdd")
        ax.title.set_color("#eeeeee")

    # ── Unit vector components ────────────────────────────────────────────────
    ax1.bar(x, stat["unit_mean"], width, yerr=stat["unit_std"],
            color=color, alpha=0.85, error_kw={"ecolor": "#ffffff", "capsize": 4})
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT_CH, rotation=30, ha="right")
    ax1.set_ylabel("Unit-vector component (mean ± std)")
    ax1.set_title(f"{emotion}  —  relative channel contribution  "
                  f"({stat['n_trials']} trial(s), {stat['n_samples']} samples)")
    uniform_uv = 1.0 / np.sqrt(N_CH)        # equal-channel unit-vector level
    ax1.set_ylim(bottom=0)
    ax1.axhline(uniform_uv, color="#ffffff", linewidth=0.8, linestyle="--",
                label=f"uniform (1/sqrt({N_CH})) = {uniform_uv:.3f}")
    ax1.legend(fontsize=7, labelcolor="#ccccdd", facecolor="#2a2a3e",
               edgecolor="#44445a")

    # ── RMS amplitude per channel ─────────────────────────────────────────────
    # We show per-channel RMS amplitude (from the raw rms before normalisation).
    # unit_mean * amp_mean gives a per-channel reconstruction of the mean RMS.
    ch_amp_mean = stat["unit_mean"] * stat["amp_mean"]
    ch_amp_std  = stat["unit_std"]  * stat["amp_mean"]   # propagated approx.

    ax2.bar(x, ch_amp_mean, width, yerr=ch_amp_std,
            color=color, alpha=0.65, error_kw={"ecolor": "#ffffff", "capsize": 4})
    ax2.set_xticks(x)
    ax2.set_xticklabels(SHORT_CH, rotation=30, ha="right")
    ax2.set_ylabel("RMS amplitude per channel (µV)")
    ax2.set_title(f"{emotion}  —  RMS amplitude  "
                  f"(overall ‖rms‖ mean={stat['amp_mean']:.1f} µV, "
                  f"std={stat['amp_std']:.1f} µV)")
    ax2.set_ylim(bottom=0)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_amplitude_comparison(stats: dict[str, dict], emotion_order: list[str]) -> str:
    """
    Single bar chart comparing mean ± std overall RMS amplitude across all
    emotions (emotions on x-axis, amplitude in µV on y-axis).
    Each bar is coloured by emotion.
    """
    emotions = [e for e in emotion_order if e in stats]
    means = np.array([stats[e]["amp_mean"] for e in emotions])
    stds  = np.array([stats[e]["amp_std"]  for e in emotions])
    colors = [EMOTION_COLORS.get(e, "#888888") for e in emotions]

    fig, ax = plt.subplots(figsize=(max(8, len(emotions) * 1.1), 4), tight_layout=True)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#2a2a3e")
    ax.tick_params(colors="#ccccdd", labelsize=9)
    ax.spines[:].set_color("#44445a")
    ax.yaxis.label.set_color("#ccccdd")
    ax.title.set_color("#eeeeee")

    x = np.arange(len(emotions))
    bars = ax.bar(x, means, 0.6, yerr=stds, color=colors, alpha=0.88,
                  error_kw={"ecolor": "#ccccdd", "capsize": 5})
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean overall RMS amplitude (µV)")
    ax.set_title("Overall RMS amplitude by emotion (hold period, all blocks/trials)")
    ax.set_ylim(bottom=0)

    # Annotate each bar with its mean value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.0f}",
            ha="center", va="bottom", fontsize=7, color="#eeeeee",
        )

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── HTML generation ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #12121f; color: #d0d0e8; font-family: 'Segoe UI', sans-serif; }
header { background: #1e1e2e; padding: 24px 32px; border-bottom: 2px solid #3a3a5c; }
header h1 { font-size: 1.5rem; color: #a0c4ff; letter-spacing: .04em; }
header p  { font-size: .85rem; color: #8888aa; margin-top: 4px; }
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px; padding: 24px 32px;
}
.meta-card {
  background: #1e1e2e; border: 1px solid #3a3a5c; border-radius: 8px;
  padding: 14px 18px;
}
.meta-card .label { font-size: .72rem; color: #8888aa; text-transform: uppercase; letter-spacing: .08em; }
.meta-card .value { font-size: 1.1rem; color: #e0e0f8; margin-top: 4px; font-weight: 600; }
.section { padding: 0 32px 32px; }
.section h2 { font-size: 1.1rem; color: #a0c4ff; margin-bottom: 16px; padding-top: 24px;
              border-top: 1px solid #3a3a5c; }
.emotion-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(680px, 1fr)); gap: 24px;
}
.emotion-card {
  background: #1e1e2e; border: 1px solid #3a3a5c; border-radius: 10px;
  overflow: hidden;
}
.emotion-card .card-title {
  padding: 10px 18px; font-size: .9rem; font-weight: 700;
  letter-spacing: .05em; color: #12121f;
}
.emotion-card img { width: 100%; display: block; }
table { border-collapse: collapse; width: 100%; font-size: .8rem; }
th, td { border: 1px solid #3a3a5c; padding: 6px 12px; text-align: left; }
th { background: #252540; color: #a0c4ff; }
tr:nth-child(even) { background: #1a1a2e; }
.pill {
  display: inline-block; border-radius: 4px; padding: 2px 8px;
  font-size: .72rem; font-weight: 700; letter-spacing: .06em; color: #12121f;
}
"""


def build_html(session_id: str, meta: dict, stats: dict[str, dict],
               emotion_plots: dict[str, str], amp_comparison_plot: str = "") -> str:

    n_samples_total = sum(s["n_samples"] for s in stats.values())
    duration_s      = meta.get("duration_s", "N/A")
    recorded_at     = meta.get("recorded_at", session_id[len(session_id)-8:])
    participant_id  = meta.get("participant_id", "–")
    session_num     = meta.get("session_number", "–")

    # ── Meta cards ────────────────────────────────────────────────────────────
    meta_cards = [
        ("Session ID",    session_id),
        ("Participant",   participant_id),
        ("Session #",     session_num),
        ("Recorded at",   recorded_at),
        ("Sample rate",   f"{FS} Hz"),
        ("Channels",      str(N_CH)),
        ("Bandpass",      f"{BP_LOW}–{BP_HIGH} Hz"),
        ("RMS window",    f"{RMS_WIN_SEC} s ({RMS_WIN_SAMP} samples)"),
        ("Hold windows",  str(sum(s["n_trials"] for s in stats.values()))),
        ("Total samples", f"{n_samples_total:,}"),
    ]
    cards_html = "\n".join(
        f'<div class="meta-card"><div class="label">{l}</div>'
        f'<div class="value">{v}</div></div>'
        for l, v in meta_cards
    )

    # ── Channel table ─────────────────────────────────────────────────────────
    ch_rows = "\n".join(
        f"<tr><td>ch{i+1}</td><td>{SHORT_CH[i]}</td><td>{CH_LABELS[i]}</td></tr>"
        for i in range(N_CH)
    )
    ch_table = (
        "<table><thead><tr>"
        "<th>Signal</th><th>Short label</th><th>Full label</th>"
        "</tr></thead><tbody>"
        f"{ch_rows}</tbody></table>"
    )

    # ── Emotion summary table ─────────────────────────────────────────────────
    emotion_order = [e for e in [
        "HAPPINESS", "ANGER", "DISGUST", "FEAR", "SURPRISE",
        "SADNESS", "CONTEMPT_LEFT", "CONTEMPT_RIGHT", "NEUTRAL",
    ] if e in stats]
    # add any extras not in the canonical order
    emotion_order += [e for e in stats if e not in emotion_order]

    summary_rows = "\n".join(
        f"<tr>"
        f'<td><span class="pill" style="background:{EMOTION_COLORS.get(e,"#666")}">{e}</span></td>'
        f"<td>{stats[e]['n_trials']}</td>"
        f"<td>{stats[e]['n_samples']:,}</td>"
        f"<td>{stats[e]['amp_mean']:.1f}</td>"
        f"<td>{stats[e]['amp_std']:.1f}</td>"
        f"</tr>"
        for e in emotion_order if e in stats
    )
    summary_table = (
        "<table><thead><tr>"
        "<th>Emotion</th><th>Trials</th><th>Samples</th>"
        "<th>‖RMS‖ mean (µV)</th><th>‖RMS‖ std (µV)</th>"
        "</tr></thead><tbody>"
        f"{summary_rows}</tbody></table>"
    )

    # ── Per-emotion plot cards ─────────────────────────────────────────────────
    emotion_cards = "\n".join(
        f'<div class="emotion-card">'
        f'<div class="card-title" style="background:{EMOTION_COLORS.get(e,"#666")}">{e}</div>'
        f'<img src="data:image/png;base64,{emotion_plots[e]}" alt="{e} plot">'
        f'</div>'
        for e in emotion_order if e in emotion_plots
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Validation Report — {session_id}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>Session Validation Report</h1>
  <p>Auto-generated · {session_id}</p>
</header>

<div class="meta-grid">{cards_html}</div>

<div class="section">
  <h2>Channels</h2>
  {ch_table}
</div>

<div class="section">
  <h2>Emotion Summary (hold periods)</h2>
  {summary_table}
</div>

<div class="section">
  <h2>RMS Amplitude Comparison Across Emotions</h2>
  <p style="font-size:.8rem; color:#8888aa; margin-bottom:16px;">
    Mean ± std of the overall 8-channel RMS amplitude magnitude (µV) during
    the hold period, aggregated across all trials and blocks.
  </p>
  <div style="background:#1e1e2e; border:1px solid #3a3a5c; border-radius:10px; overflow:hidden; max-width:900px;">
    <img src="data:image/png;base64,{amp_comparison_plot}" alt="Amplitude comparison" style="width:100%;display:block;">
  </div>
</div>

<div class="section">
  <h2>Feature Distributions by Emotion (hold period, all blocks/trials)</h2>
  <p style="font-size:.8rem; color:#8888aa; margin-bottom:16px;">
    Top subplot: mean ± std of per-sample unit-vector components (relative channel
    contribution). The dashed line marks the uniform-contribution level
    1/sqrt({N_CH}) = {1/N_CH**0.5:.3f} — the value each component would have if
    all 8 channels were equally activated. Values above it indicate above-average
    relative contribution; values below indicate below-average.<br>
    Bottom subplot: reconstructed mean ± std of per-channel RMS amplitude (µV).
  </p>
  <div class="emotion-grid">{emotion_cards}</div>
</div>

</body>
</html>
"""


# ── Session discovery & report entry point ─────────────────────────────────────

def find_sessions(data_dir: Path, session_filter: str | None = None) -> list[Path]:
    """Return subdirectories of data_dir that contain an _EXG.csv file."""
    sessions = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if session_filter and d.name != session_filter:
            continue
        exg_files = list(d.glob("*_EXG.csv"))
        if exg_files:
            sessions.append(d)
    return sessions


def load_meta(session_dir: Path) -> dict:
    meta_files = list(session_dir.glob("*_META.json"))
    if meta_files:
        with open(meta_files[0], encoding="utf-8") as f:
            return json.load(f)
    return {}


def process_session(session_dir: Path) -> None:
    session_id = session_dir.name
    logger.info("Generating report for session: %s", session_id)

    # Paths
    exg_path     = next(session_dir.glob("*_EXG.csv"))
    markers_path = next(session_dir.glob("*_MARKERS.csv"))
    report_path  = session_dir / f"{session_id}_REPORT.html"

    # Load
    exg = pd.read_csv(exg_path)
    logger.info("  EXG loaded: %d rows", len(exg))

    markers = pd.read_csv(markers_path)
    logger.info("  Markers loaded: %d events", len(markers))

    meta = load_meta(session_dir)

    # Feature pipeline
    logger.info("  Bandpass filtering and computing rolling RMS ...")
    ts, unit_vec, rms_amp = extract_features(exg)

    # Hold windows
    windows = parse_hold_windows(markers)
    logger.info("  Hold windows found: %d", len(windows))

    # Statistics
    stats = compute_emotion_stats(ts, unit_vec, rms_amp, windows)
    logger.info("  Emotions: %s", ", ".join(stats.keys()))

    # Per-emotion plots
    logger.info("  Generating per-emotion plots ...")
    emotion_plots: dict[str, str] = {}
    for emotion, stat in stats.items():
        color = EMOTION_COLORS.get(emotion, "#888888")
        emotion_plots[emotion] = plot_emotion(emotion, stat, color)

    # Amplitude comparison plot
    emotion_order_for_plot = [e for e in [
        "HAPPINESS", "ANGER", "DISGUST", "FEAR", "SURPRISE",
        "SADNESS", "CONTEMPT_LEFT", "CONTEMPT_RIGHT", "NEUTRAL",
    ] if e in stats]
    emotion_order_for_plot += [e for e in stats if e not in emotion_order_for_plot]
    amp_plot = plot_amplitude_comparison(stats, emotion_order_for_plot)

    # HTML
    html = build_html(session_id, meta, stats, emotion_plots, amp_comparison_plot=amp_plot)
    report_path.write_text(html, encoding="utf-8")
    logger.info("  Report written: %s", report_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a validation HTML report for recorded sessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="Process only this session (folder name under data/). "
             "Defaults to all sessions found in data/.",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        default=str(DATA_DIR),
        help="Root data directory.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    data_dir = Path(args.data_dir)

    sessions = find_sessions(data_dir, session_filter=args.session)
    if not sessions:
        print(f"No sessions found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    for session_dir in sessions:
        process_session(session_dir)

    logger.info("All done.")


if __name__ == "__main__":
    main()
