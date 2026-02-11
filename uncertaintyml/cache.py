"""
Thread-safe LRU prediction cache with TTL expiry.

Provides sub-millisecond response for repeated identical predictions,
avoiding redundant MC Dropout forward passes.
"""

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CacheStats:
    """Observable cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class PredictionCache:
    """
    Thread-safe LRU cache with TTL for prediction results.

    Features:
        - Deterministic hashing of feature dicts
        - LRU eviction when max_size reached
        - TTL-based expiry (default 1 hour)
        - Thread-safe via threading.Lock
        - Observable stats (hits, misses, evictions)

    Usage:
        cache = PredictionCache(max_size=10000, ttl_seconds=3600)
        key = cache.make_key({"Age": 55, "RestingBP": 140})

        result = cache.get(key)
        if result is None:
            result = model.predict(...)
            cache.put(key, result)
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    @staticmethod
    def make_key(features: Dict[str, Any]) -> str:
        """Generate deterministic cache key from feature dict."""
        sorted_items = sorted(features.items())
        canonical = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached prediction. Returns None on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.stats.misses += 1
                return None

            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                self.stats.misses += 1
                return None

            self._cache.move_to_end(key)
            self.stats.hits += 1
            return copy.deepcopy(entry["value"])

    def put(self, key: str, value: Dict) -> None:
        """Store prediction result in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = {"value": value, "timestamp": time.time()}
                return

            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self.stats.evictions += 1

            self._cache[key] = {"value": value, "timestamp": time.time()}

    def invalidate(self) -> None:
        """Clear entire cache (e.g., after model retrain)."""
        with self._lock:
            self._cache.clear()
            self.stats = CacheStats()

    @property
    def size(self) -> int:
        return len(self._cache)
