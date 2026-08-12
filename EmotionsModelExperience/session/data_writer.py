"""
Data writers for the Emotions Model Experience.

Each session gets its own subfolder:
  {DATA_DIR}/HERMES_{PID}_S{SESSION}_{YYYYMMDD}/
    HERMES_{PID}_S{SESSION}_{YYYYMMDD}_EXG.csv
    HERMES_{PID}_S{SESSION}_{YYYYMMDD}_MARKERS.csv
    HERMES_{PID}_S{SESSION}_{YYYYMMDD}_META.json
    HERMES_{PID}_S{SESSION}_{YYYYMMDD}_VIDEO.webm
"""

import csv
import json
import threading
import queue
import os
from datetime import datetime
from typing import Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _build_stem(pid: str, session: int, date_str: str) -> str:
    return f"HERMES_{pid}_S{session:02d}_{date_str}"


# ── EXG writer ────────────────────────────────────────────────────────────────

class EMGWriter:
    """
    Thread-safe EXG (EMG/EEG) sample writer.

    Runs a dedicated background thread that drains a queue and writes rows
    to CSV so the BLE callback thread is never blocked by disk I/O.

    Each row: timestamp, ch1, ch2, ..., ch8
    NaN values are written as empty cells (compatible with numpy.genfromtxt).
    """

    _SENTINEL = object()

    def __init__(self, pid: str, session: int, date_str: str, data_dir: str):
        stem = _build_stem(pid, session, date_str)
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, f"{stem}_EXG.csv")

        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._path

    def write_samples(self, samples: list[list[float]], timestamp_epoch: float) -> None:
        """
        Enqueue a list of samples (one per ADS1299 conversion step).
        Each sample is a list of 8 µV floats (NaN for dropped packets).
        Timestamps are interpolated at 1/SAMPLE_RATE spacing starting from
        timestamp_epoch, which is the arrival time of the BLE packet.
        """
        dt = 1.0 / config.SAMPLE_RATE
        n = len(samples)
        # Back-date: first sample is n*dt before packet arrival
        t0 = timestamp_epoch - (n - 1) * dt
        rows = [(t0 + i * dt, *s) for i, s in enumerate(samples)]
        self._q.put(rows)

    def close(self) -> None:
        """Flush remaining data and close the file."""
        self._q.put(self._SENTINEL)
        self._thread.join()

    # ── internals ─────────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        with open(self._path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp"] + [f"ch{i+1}" for i in range(config.N_CHANNELS)]
            writer.writerow(header)
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
                for row in item:
                    writer.writerow(
                        [f"{row[0]:.6f}"] + [
                            "" if (isinstance(v, float) and v != v) else f"{v:.4f}"
                            for v in row[1:]
                        ]
                    )


# ── Marker writer ─────────────────────────────────────────────────────────────

class MarkerWriter:
    """
    Synchronous marker writer.  Writes one row per event immediately.
    Must be called from the orchestrator's asyncio thread (or protected by a lock
    if called from multiple threads).

    Each row: timestamp, marker_string
    """

    def __init__(self, pid: str, session: int, date_str: str, data_dir: str):
        stem = _build_stem(pid, session, date_str)
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, f"{stem}_MARKERS.csv")
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "marker"])

    @property
    def path(self) -> str:
        return self._path

    def write(self, marker: str) -> None:
        ts = datetime.now().timestamp()
        self._writer.writerow([f"{ts:.6f}", marker])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ── Meta writer ───────────────────────────────────────────────────────────────

class MetaWriter:
    """
    Collects session metadata and writes META.json at the end of the session.
    Fields match the protocol spec (§7).
    """

    def __init__(self, pid: str, session: int, date_str: str, data_dir: str):
        stem = _build_stem(pid, session, date_str)
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, f"{stem}_META.json")
        self._meta: dict[str, Any] = {
            "participant_id":  pid,
            "session":         session,
            "date":            date_str,
            "sample_rate_hz":  config.SAMPLE_RATE,
            "n_channels":      config.N_CHANNELS,
            "channel_labels":  config.CHANNEL_LABELS,
            "blocks":          [],
            "exg_file":        f"{stem}_EXG.csv",
            "markers_file":    f"{stem}_MARKERS.csv",
        }

    @property
    def path(self) -> str:
        return self._path

    def set_experimenter(self, experimenter_id: str) -> None:
        self._meta["experimenter_id"] = experimenter_id

    def set_device_mac(self, mac: str) -> None:
        self._meta["device_mac"] = mac

    def add_block(
        self,
        block_number: int,
        mirror: bool,
        trial_order: list[str],
        rng_seed: int,
    ) -> None:
        self._meta["blocks"].append({
            "block_number": block_number,
            "mirror":       mirror,
            "trial_order":  trial_order,
            "rng_seed":     rng_seed,
        })

    def set_consent(
        self,
        first_name: str,
        last_name:  str,
        email:      str,
        lang:       str,
        timestamp:  float,
    ) -> None:
        self._meta["consent"] = {
            "first_name": first_name,
            "last_name":  last_name,
            "email":      email,
            "lang":       lang,
            "timestamp":  timestamp,
        }

    def set_notes(self, notes: str) -> None:
        self._meta["notes"] = notes

    def set_channel_quality(self, quality: dict[str, str]) -> None:
        """
        quality: mapping channel label → "good" | "poor" | "noisy" | "flat"
        """
        self._meta["channel_quality"] = quality

    def set_session_times(self, start_epoch: float, end_epoch: float) -> None:
        self._meta["session_start_epoch"] = start_epoch
        self._meta["session_end_epoch"]   = end_epoch
        self._meta["duration_s"]          = round(end_epoch - start_epoch, 2)

    def write(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)


# ── Video writer ──────────────────────────────────────────────────────────────

class VideoWriter:
    """
    Receives raw WebM binary chunks from the browser's MediaRecorder and
    appends them sequentially to a single .webm file.

    Time-alignment is achieved by the VIDEO_START marker written to
    MARKERS.csv at the epoch timestamp the browser sends when recording begins.
    """

    def __init__(self, pid: str, session: int, date_str: str, session_dir: str):
        stem = _build_stem(pid, session, date_str)
        self._path = os.path.join(session_dir, f"{stem}_VIDEO.webm")
        self._file = open(self._path, "wb")
        self._bytes_written = 0

    @property
    def path(self) -> str:
        return self._path

    def write_chunk(self, chunk: bytes) -> None:
        self._file.write(chunk)
        self._bytes_written += len(chunk)

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        logger.info("Video saved: %s  (%.1f KB)", self._path, self._bytes_written / 1024)


import logging as _logging
logger = _logging.getLogger("data_writer")


# ── Convenience factory ───────────────────────────────────────────────────────

def create_writers(pid: str, session: int, data_dir: str = config.DATA_DIR):
    """
    Return (EMGWriter, MarkerWriter, MetaWriter, VideoWriter) for a new session.
    Files are placed inside a dedicated subfolder:
      {data_dir}/HERMES_{pid}_S{session:02d}_{date}/
    """
    date_str    = datetime.now().strftime("%Y%m%d")
    stem        = _build_stem(pid, session, date_str)
    session_dir = os.path.join(data_dir, stem)
    os.makedirs(session_dir, exist_ok=True)

    emg     = EMGWriter(pid, session, date_str, session_dir)
    markers = MarkerWriter(pid, session, date_str, session_dir)
    meta    = MetaWriter(pid, session, date_str, session_dir)
    video   = VideoWriter(pid, session, date_str, session_dir)
    return emg, markers, meta, video
