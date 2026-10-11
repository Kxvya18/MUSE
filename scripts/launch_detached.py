"""Start scripts/pipeline.sh in its own session (survives the launching app/terminal),
under caffeinate so the Mac does not sleep while it runs."""
import os, subprocess, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.fork() == 0:
    os.setsid()
    subprocess.Popen(["caffeinate", "-is", "bash", os.path.join(root, "scripts", "pipeline.sh")],
                     cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    os._exit(0)
