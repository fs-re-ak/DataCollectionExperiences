"""
Session orchestrator for the Emotions Model Experience.

Runs a FastAPI server that:
  • Serves the static web/ directory (participant display)
  • Exposes a WebSocket endpoint at /ws for real-time stimulus control
  • Exposes GET /api/browse-dir  for native folder picker
  • Runs the full session state machine as an asyncio background task

Lifecycle
─────────
  1. Server starts with no session info (no writers, no proxy).
  2. Browser opens → registration screen shown.
  3. Experimenter may trigger BLE scan (WS: scan_request).
  4. Experimenter submits registration form (WS: register {pid, session,
     experimenter, data_dir, mac}).
  5. Server creates writers, generates block orders, connects AlchemiacProxy.
  6. Server sends WS: registered → browser shows waiting screen.
  7. Experimenter clicks Démarrer (WS: start) → session state machine runs.

State machine sequence (after registration)
────────────────────────────────────────────
  WAITING_FOR_START
  → SESSION_START
  → BASELINE_PRE
  → Block × 3 (BLOCK_START → trials → BLOCK_END → REST)
  → BASELINE_POST
  → SESSION_END
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from session.data_writer import EMGWriter, MarkerWriter, MetaWriter, VideoWriter, create_writers
from session.randomizer import BlockOrder, build_session_order

logger = logging.getLogger("orchestrator")

WEB_DIR    = Path(__file__).resolve().parent.parent / "web"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _generate_report_bg(session_dir: Path) -> None:
    """
    Run the HTML report generator in a background thread after session
    completion.  Failures are logged as warnings so they never affect the
    session runner.
    """
    try:
        from generate_report import process_session
        process_session(session_dir)
        logger.info("Validation report written to %s", session_dir)
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)


def _make_eeg_callback(emg_writer: EMGWriter):
    def callback(samples):
        emg_writer.write_samples(samples, time.time())
    return callback


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SessionOrchestrator:
    """
    Holds all server-side session state and wires together the FastAPI app,
    WebSocket connection, data writers, and the async state machine.

    Writers, proxy, and session identity are all deferred until the
    registration form is submitted from the browser.
    """

    def __init__(self, dry_run: bool = False, seed_override: Optional[int] = None):
        self.dry_run        = dry_run
        self._seed_override = seed_override

        # Session identity — filled in after registration
        self.pid:           Optional[str]              = None
        self.session:       Optional[int]              = None
        self.block_orders:  Optional[list[BlockOrder]] = None
        self.emg_writer:    Optional[EMGWriter]        = None
        self.marker_writer: Optional[MarkerWriter]     = None
        self.meta_writer:   Optional[MetaWriter]       = None
        self.video_writer:  Optional[VideoWriter]      = None
        self._proxy                                    = None
        self._ready         = False   # True once registration is complete

        # Experimenter control
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._abort_event = asyncio.Event()
        self._session_task: Optional[asyncio.Task] = None

        # WebSocket
        self._ws: Optional[WebSocket] = None
        self._ws_lock = asyncio.Lock()

        # Live status broadcast
        self._status: dict = {
            "state":     "registration",
            "block":     0,
            "trial":     0,
            "n_trials":  len(config.EMOTIONS_BASE),
            "n_blocks":  config.N_BLOCKS,
            "paused":    False,
            "elapsed_s": 0,
        }
        self._session_start: float = 0.0

        # Build FastAPI app
        self.app = FastAPI(title="Emotions Model Experience")
        self._register_routes()

    # ── FastAPI routes ────────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/api/browse-dir")
        async def browse_dir():
            """
            Opens a native OS folder-picker dialog on the server machine and
            returns the selected path.  Falls back to config.DATA_DIR on cancel.
            """
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes("-topmost", True)
                path = filedialog.askdirectory(
                    title="Sélectionner le dossier de données",
                    initialdir=os.path.abspath(config.DATA_DIR),
                )
                root.destroy()
                return JSONResponse({"path": path if path else config.DATA_DIR})
            except Exception as e:
                logger.warning("browse-dir failed: %s", e)
                return JSONResponse({"path": config.DATA_DIR})

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            async with self._ws_lock:
                if self._ws is not None:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "msg":  "Another client is already connected.",
                    }))
                    await ws.close()
                    return
                self._ws = ws
            logger.info("WebSocket client connected")
            try:
                # Tell browser whether BLE scanning is available
                await self._send({
                    "type":    "server_info",
                    "dry_run": self.dry_run,
                })
                await self._push_status()
                await self._ws_receive_loop()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            finally:
                async with self._ws_lock:
                    self._ws = None

        # Assets (images, etc.) — must be mounted before the catch-all "/"
        if ASSETS_DIR.is_dir():
            app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

        # Static files — must be last
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    async def _send(self, msg: dict) -> None:
        if self._ws is not None:
            try:
                await self._ws.send_text(json.dumps(msg))
            except Exception as e:
                logger.warning("WebSocket send failed: %s", e)

    async def _push_status(self) -> None:
        self._status["elapsed_s"] = (
            round(time.time() - self._session_start, 1)
            if self._session_start else 0
        )
        await self._send({"type": "status", **self._status})

    async def _ws_receive_loop(self) -> None:
        """Handle incoming messages from the browser (text JSON and binary video chunks)."""
        while True:
            frame = await self._ws.receive()

            if frame["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            # Binary frame → video chunk
            if frame.get("bytes"):
                if self.video_writer is not None:
                    self.video_writer.write_chunk(frame["bytes"])
                continue

            raw = frame.get("text", "")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type", "")

            # ── Registration phase ────────────────────────────────────────────
            if mtype == "scan_request":
                asyncio.create_task(self._run_scan())

            elif mtype == "register":
                asyncio.create_task(self._handle_register(msg))

            # ── Consent (must arrive before start) ───────────────────────────
            elif mtype == "consent":
                if self.meta_writer:
                    self.meta_writer.set_consent(
                        first_name=msg.get("first_name", ""),
                        last_name =msg.get("last_name",  ""),
                        email     =msg.get("email",      ""),
                        lang      =msg.get("lang",       "fr"),
                        timestamp =msg.get("timestamp",  time.time()),
                    )

            # ── Session control (only valid after registration) ───────────────
            elif mtype == "start":
                if not self._ready:
                    await self._send({"type": "error", "msg": "Session not registered yet."})
                elif self._session_task is None or self._session_task.done():
                    self._session_task = asyncio.create_task(self._run_session())

            elif mtype == "pause":
                if self.marker_writer:
                    self._pause_event.clear()
                    self._status["paused"] = True
                    self.marker_writer.write("SESSION_PAUSE")
                    await self._push_status()

            elif mtype == "resume":
                if self.marker_writer:
                    self._pause_event.set()
                    self._status["paused"] = False
                    self.marker_writer.write("SESSION_RESUME")
                    await self._push_status()

            elif mtype == "abort":
                if self.marker_writer:
                    self._abort_event.set()
                    self._pause_event.set()
                    self.marker_writer.write("SESSION_ABORT")
                    await self._push_status()

            elif mtype == "ack":
                if self.marker_writer:
                    emotion = msg.get("emotion", "?")
                    block   = msg.get("block", 0)
                    trial   = msg.get("trial", 0)
                    self.marker_writer.write(f"STIM_ACK_{emotion}_B{block}_T{trial}")

            elif mtype == "channel_quality":
                if self.meta_writer:
                    self.meta_writer.set_channel_quality(msg.get("quality", {}))

            elif mtype == "notes":
                if self.meta_writer:
                    self.meta_writer.set_notes(msg.get("notes", ""))

            elif mtype == "video_start":
                # Browser sends the epoch timestamp when MediaRecorder actually starts.
                # Write it as a marker so EEG and video share a common time reference.
                ts = msg.get("timestamp", time.time())
                if self.marker_writer:
                    self.marker_writer.write(f"VIDEO_START_T{ts:.6f}")
                logger.info("Video recording started at %.6f", ts)

            elif mtype == "video_stop":
                if self.video_writer:
                    self.video_writer.close()
                    self.video_writer = None
                logger.info("Video recording stopped")

    # ── BLE scan ──────────────────────────────────────────────────────────────

    async def _run_scan(self) -> None:
        if self.dry_run:
            await self._send({"type": "scan_not_available"})
            return
        try:
            from bleak import BleakScanner
            await self._send({"type": "scan_started"})
            devices = await BleakScanner.discover(timeout=5.0)
            results = [
                {"name": d.name, "address": d.address}
                for d in devices
                if d.name and "hermes" in d.name.lower()
            ]
            results.sort(key=lambda d: d["name"].lower())
            await self._send({"type": "scan_results", "devices": results})
        except Exception as e:
            logger.error("BLE scan failed: %s", e)
            await self._send({"type": "scan_error", "msg": str(e)})

    # ── Registration ──────────────────────────────────────────────────────────

    async def _handle_register(self, msg: dict) -> None:
        """
        Called when the browser submits the registration form.
        Creates writers, generates block order, connects the proxy.
        """
        if self._ready:
            await self._send({"type": "error", "msg": "Session already registered."})
            return

        pid          = msg.get("pid", "").strip()
        session      = int(msg.get("session", 1))
        experimenter = msg.get("experimenter", "").strip()
        data_dir     = msg.get("data_dir", config.DATA_DIR).strip() or config.DATA_DIR
        mac          = msg.get("mac", None)

        if not pid:
            await self._send({"type": "register_error", "msg": "PID is required."})
            return

        # ── Writers ───────────────────────────────────────────────────────────
        try:
            self.emg_writer, self.marker_writer, self.meta_writer, self.video_writer = \
                create_writers(pid=pid, session=session, data_dir=data_dir)
        except Exception as e:
            await self._send({"type": "register_error", "msg": f"Could not create output files: {e}"})
            return

        # ── Block order ───────────────────────────────────────────────────────
        self.block_orders = build_session_order(seed_override=self._seed_override)

        # ── Meta ──────────────────────────────────────────────────────────────
        if experimenter:
            self.meta_writer.set_experimenter(experimenter)
        if mac:
            self.meta_writer.set_device_mac(mac)
        for b in self.block_orders:
            self.meta_writer.add_block(b.block_number, b.mirror, b.trial_order, b.rng_seed)
            logger.info(
                "Block %d (mirror=%s): %s",
                b.block_number, b.mirror, " → ".join(b.trial_order),
            )

        # Store identity
        self.pid     = pid
        self.session = session
        self._status["pid"]     = pid
        self._status["session"] = session

        # ── BLE connection ────────────────────────────────────────────────────
        if not self.dry_run:
            if not mac:
                await self._send({"type": "register_error", "msg": "No device selected."})
                return
            connected = await self._connect_device(mac)
            if not connected:
                return   # connect_error already sent

        logger.info(
            "Session registered — PID=%s S%02d  EMG→%s  Markers→%s",
            pid, session, self.emg_writer.path, self.marker_writer.path,
        )

        self._ready = True
        self._status["state"] = "waiting"

        await self._send({
            "type":        "registered",
            "pid":         pid,
            "session":     session,
            "emg_path":    self.emg_writer.path,
            "markers_path":self.marker_writer.path,
            "meta_path":   self.meta_writer.path,
            "video_path":  self.video_writer.path,
        })
        await self._push_status()

    # ── BLE connect ───────────────────────────────────────────────────────────

    async def _connect_device(self, mac: str) -> bool:
        """
        Instantiates AlchemiacProxy and waits for connection in an executor
        so the asyncio loop is not blocked.  Returns True on success.
        """
        await self._send({"type": "connecting", "mac": mac})
        try:
            from Devices.AlchemiacProxy import AlchemiacProxy
        except ImportError:
            await self._send({"type": "connect_error", "msg": "Cannot import AlchemiacProxy."})
            return False

        try:
            self._proxy = AlchemiacProxy(
                mac_address=mac,
                eeg_callback=_make_eeg_callback(self.emg_writer),
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._proxy.waitForConnected(timeout=AlchemiacProxy.CONNECT_TIMEOUT),
            )
            await self._send({"type": "connected", "mac": mac})
            logger.info("BLE connected to %s", mac)
            return True
        except Exception as e:
            logger.error("BLE connection failed: %s", e)
            await self._send({"type": "connect_error", "msg": str(e)})
            self._proxy = None
            return False

    # ── Interruptible sleep ───────────────────────────────────────────────────

    async def _sleep(self, duration: float, granularity: float = 0.1) -> bool:
        """
        Sleep for `duration` seconds, respecting pause and abort signals.
        Returns False if the session was aborted, True otherwise.
        """
        deadline         = time.monotonic() + duration
        last_status_push = time.monotonic()

        while True:
            await self._pause_event.wait()
            if self._abort_event.is_set():
                return False

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            await asyncio.sleep(min(granularity, remaining))

            if time.monotonic() - last_status_push >= 1.0:
                await self._push_status()
                last_status_push = time.monotonic()

        return True

    # ── State machine ─────────────────────────────────────────────────────────

    async def _run_session(self) -> None:
        self._session_start = time.time()
        self.meta_writer.set_session_times(self._session_start, 0)

        try:
            await self._phase_session_start()
            if self._abort_event.is_set():
                return

            await self._phase_baseline("PRE")
            if self._abort_event.is_set():
                return

            for b_idx, block in enumerate(self.block_orders):
                ok = await self._phase_block(block)
                if not ok or self._abort_event.is_set():
                    return
                if b_idx < len(self.block_orders) - 1:
                    ok = await self._phase_rest(block.block_number)
                    if not ok:
                        return

            await self._phase_baseline("POST")
            await self._phase_session_end()

        finally:
            end_time = time.time()
            self.meta_writer.set_session_times(self._session_start, end_time)
            self.meta_writer.write()
            self.emg_writer.close()
            self.marker_writer.close()
            if self.video_writer is not None:
                try:
                    self.video_writer.close()
                except Exception:
                    pass
                self.video_writer = None
            if self._proxy is not None:
                try:
                    self._proxy.disconnect()
                except Exception:
                    pass
            logger.info("Session complete. Files written.")

            # Generate validation report in the background — does not block
            # the server and never raises to the caller.
            if self.emg_writer is not None:
                session_dir = Path(self.emg_writer.path).parent
                threading.Thread(
                    target=_generate_report_bg,
                    args=(session_dir,),
                    name="report-generator",
                    daemon=False,   # let it finish even if server exits
                ).start()

    async def _phase_session_start(self) -> None:
        self._write_marker("SESSION_START")
        self._status["state"] = "running"
        await self._send({"type": "session_start"})
        await self._push_status()

    async def _phase_baseline(self, which: str) -> None:
        self._write_marker(f"BASELINE_{which}_START")
        self._status["state"] = f"baseline_{which.lower()}"
        duration = config.T_BASELINE_PRE if which == "PRE" else config.T_BASELINE_POST
        await self._send({"type": "baseline", "which": which, "duration": duration})
        await self._push_status()
        await self._sleep(duration)
        self._write_marker(f"BASELINE_{which}_END")

    async def _phase_block(self, block: BlockOrder) -> bool:
        bn          = block.block_number
        mirror_flag = 1 if block.mirror else 0

        self._write_marker(f"BLOCK_START_B{bn}_M{mirror_flag}")
        self._status["block"] = bn
        self._status["state"] = "block"
        await self._send({
            "type":     "block_start",
            "block":    bn,
            "mirror":   block.mirror,
            "n_trials": len(block.trial_order),
            "trials":   block.trial_order,
        })
        await self._push_status()

        for t_idx, emotion in enumerate(block.trial_order):
            ok = await self._phase_trial(bn, t_idx + 1, emotion, block.mirror)
            if not ok:
                return False

        self._write_marker(f"BLOCK_END_B{bn}_M{mirror_flag}")
        await self._send({"type": "block_end", "block": bn})
        return True

    async def _phase_trial(self, block_num: int, trial_num: int, emotion: str, mirror: bool) -> bool:
        """
        Trial sequence (participant sees a single uninterrupted expression window):

          t=0                   STIM marker — instruction shown, countdown starts
          t=T_FIXATION          HOLD marker — participant is now mid-expression
                                (display unchanged, countdown continues)
          t=T_FIXATION+T_HOLD   RELAX marker — relax screen shown
          t=T_FIXATION+T_HOLD+T_RELAX  — next trial
        """
        self._status["trial"] = trial_num
        display_duration = config.T_FIXATION + config.T_HOLD  # full visible expression window

        # ── STIM: show instruction immediately, start countdown ───────────────
        side = (
            "LEFT"  if emotion == "CONTEMPT_LEFT"  else
            "RIGHT" if emotion == "CONTEMPT_RIGHT" else
            None
        )
        self._write_marker(f"STIM_{emotion}_B{block_num}_T{trial_num}")
        await self._send({
            "type":           "stimulus",
            "emotion":        emotion,
            "label_fr":       config.EMOTION_LABELS_FR.get(emotion, emotion),
            "label_en":       config.EMOTION_LABELS_EN.get(emotion, emotion),
            "instruction_fr": config.EMOTION_INSTRUCTIONS_FR.get(emotion, ""),
            "instruction_en": config.EMOTION_INSTRUCTIONS_EN.get(emotion, ""),
            "side":           side,
            "mirror":         mirror,
            "block":          block_num,
            "trial":          trial_num,
            "duration":       display_duration,
        })
        await self._push_status()

        # ── Setup phase (instruction reading) ─────────────────────────────────
        if not await self._sleep(config.T_FIXATION):
            return False

        # ── HOLD: marker-only, no display change ──────────────────────────────
        self._write_marker(f"HOLD_{emotion}_B{block_num}_T{trial_num}")

        # ── Hold phase (expression maintained) ────────────────────────────────
        if not await self._sleep(config.T_HOLD):
            return False

        # ── RELAX ─────────────────────────────────────────────────────────────
        self._write_marker(f"RELAX_{emotion}_B{block_num}_T{trial_num}")
        await self._send({"type": "relax", "block": block_num, "trial": trial_num, "duration": config.T_RELAX})
        if not await self._sleep(config.T_RELAX):
            return False

        return True

    async def _phase_rest(self, last_block_num: int) -> bool:
        self._write_marker(f"REST_START_B{last_block_num}")
        self._status["state"] = "rest"
        await self._send({"type": "rest", "duration": config.T_REST})
        await self._push_status()
        ok = await self._sleep(config.T_REST)
        self._write_marker(f"REST_END_B{last_block_num}")
        return ok

    async def _phase_session_end(self) -> None:
        self._write_marker("SESSION_END")
        self._status["state"] = "done"
        await self._send({"type": "session_end"})
        await self._push_status()

    def _write_marker(self, marker: str) -> None:
        if self.marker_writer:
            self.marker_writer.write(marker)
        logger.debug("MARKER: %s", marker)
