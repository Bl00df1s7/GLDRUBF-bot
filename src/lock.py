"""PID lock preventing overlapping bot runs."""

import os
from contextlib import contextmanager


@contextmanager
def process_lock(path: str = "/tmp/gldrubf.lock"):
    """Acquire an exclusive PID lock and remove it when the run ends."""
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                with open(path) as file:
                    pid = int(file.read().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                os.unlink(path)
                continue
            raise RuntimeError(f"Another bot process is running: PID {pid}")

    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
