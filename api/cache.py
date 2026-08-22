"""Thread-safe in-memory LRU cache: normalized-text hash -> (label, conf, version)."""
import hashlib
import threading
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self._data: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key_for(text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:32]

    def get(self, key: str):
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def set(self, key: str, value: tuple) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
