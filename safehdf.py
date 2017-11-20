import os
import time

from pandas import HDFStore


class SafeHDF5Store(HDFStore):
    """Implement safe HDFStore by obtaining file lock. Multiple writes will queue if lock is not obtained."""

    def __init__(self, *args, **kwargs):
        """Initialize and obtain file lock."""

        interval = kwargs.pop('probe_interval', 1)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (IOError, OSError):
                time.sleep(interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        """Exit and remove file lock."""

        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)
