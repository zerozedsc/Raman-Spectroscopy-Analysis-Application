import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartHandler(FileSystemEventHandler):
    def __init__(self):
        self.restart_triggered = False

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            self.restart_triggered = True

def run_app():
    # The command to run your app
    cmd = ["uv", "run", "main.py"]
    
    handler = RestartHandler()
    observer = Observer()
    observer.schedule(handler, path=".", recursive=True)
    observer.start()

    process = subprocess.Popen(cmd)
    print("🚀 App started. Watching for changes...")

    try:
        while True:
            # 1. Check if code changed
            if handler.restart_triggered:
                print("\n♻️  Change detected. Restarting...")
                process.terminate()  # Kill current app
                process.wait()       # Wait for cleanup
                process = subprocess.Popen(cmd) # Start new
                handler.restart_triggered = False

            # 2. Check if App exited naturally (User clicked X)
            retcode = process.poll()
            if retcode is not None:
                if retcode == 0:
                    print("✅ App closed cleanly. Exiting watcher.")
                    observer.stop()
                    sys.exit(0)
                else:
                    # App crashed. Keep watching for fixes.
                    pass 

            time.sleep(0.5)
            
    except KeyboardInterrupt:
        observer.stop()
        process.terminate()

    observer.join()

if __name__ == "__main__":
    run_app()