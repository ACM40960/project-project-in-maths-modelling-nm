import subprocess
import time
import signal
import os
import sys

def launch(cmd, name):
    print(f"\n🚀 Launching {name}...")
    return subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

def main():
    procs = []
    try:
        # 1. FastAPI server
        procs.append(launch(["uvicorn", "src.banksim_fraud.api:app", "--reload"], "FastAPI API"))

        time.sleep(2)

        # 2. Streamer
        procs.append(launch(["python", "tools/stream_and_score.py"], "Streamer"))

        time.sleep(1)

        # 3. Streamlit dashboard (headless mode = no browser)
        procs.append(launch([
            "streamlit", "run", "tools/dashboard.py",
            "--server.headless", "true"
        ], "Streamlit Dashboard"))

        print("\n✅ All systems running. Press Ctrl+C to stop everything.\n")

        # Keep running until user interrupts
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected. Shutting down all processes...\n")

    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("✅ All processes terminated. Demo stopped.\n")

if __name__ == "__main__":
    main()
