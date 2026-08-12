"""
run_session.py — Entry point for the Emotions Model Experience.

Usage
─────
    # Normal lab use — registration is done in the browser
    python run_session.py

    # Dry-run (no BLE device — for UI testing)
    python run_session.py --no-device

Options
───────
    --port         HTTP/WS server port (default: 8765)
    --no-browser   Don't auto-open the browser
    --no-device    Skip BLE connection (dry run, EMG file will be empty)
    --seed         Override master RNG seed (integer, for exact reproducibility)

Participant ID, session number, experimenter ID, data directory, and BLE MAC
address are all collected via the registration form in the browser.
"""

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser

import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
import config
from session.orchestrator import SessionOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_session")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emotions Model Experience — session runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",       type=int, default=config.SERVER_PORT, help="Web server port")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    p.add_argument("--no-device",  action="store_true", help="Skip BLE — dry run")
    p.add_argument("--seed",       type=int, default=None, help="RNG seed override")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    orchestrator = SessionOrchestrator(
        dry_run=args.no_device,
        seed_override=args.seed,
    )

    if args.no_device:
        logger.warning("--no-device: BLE disabled. EMG file will be empty.")

    uv_config = uvicorn.Config(
        app=orchestrator.app,
        host=config.SERVER_HOST,
        port=args.port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(uv_config)

    if not args.no_browser:
        url = f"http://{config.SERVER_HOST}:{args.port}"

        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(url)
            logger.info("Browser opened at %s", url)

        threading.Thread(target=_open_browser, daemon=True).start()

    logger.info(
        "Server starting at http://%s:%d  (fill in the registration form, then press F1 for experimenter panel)",
        config.SERVER_HOST, args.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        logger.info("Done.")


if __name__ == "__main__":
    main()
