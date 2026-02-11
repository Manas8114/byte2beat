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
    errors: int = 0  # Track Redis connection errors

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
            "errors": self.errors,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheBackend:
    """Abstract base for cache storage backends."""
    def get(self, key: str) -> Optional[Dict]: ...
    def put(self, key: str, value: Dict, ttl: int) -> None: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...
    def close(self) -> None: ...


class MemoryBackend(CacheBackend):
    """Local thread-safe LRU cache."""
    def __init__(self, max_size: int):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            # Check TTL
            if time.time() > entry["expiry"]:
                del self._cache[key]
                return None

            self._cache.move_to_end(key)
            return copy.deepcopy(entry["value"])

    def put(self, key: str, value: Dict, ttl: int) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                # We don't track evictions inside backend, clearer in wrapper
            
            self._cache[key] = {
                "value": value,
                "expiry": time.time() + ttl
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class RedisBackend(CacheBackend):
    """Production-grade Redis cache."""
    def __init__(self, url: str):
        try:
            import redis
            self.client = redis.from_url(url, socket_connect_timeout=1)
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package. Run: pip install redis")

    def get(self, key: str) -> Optional[Dict]:
        try:
            val = self.client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def put(self, key: str, value: Dict, ttl: int) -> None:
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self.client.flushdb()
        except Exception:
            pass

    def size(self) -> int:
        try:
            return self.client.dbsize()
        except Exception:
            return 0


class PredictionCache:
    """
    Facade for prediction caching. defaults to MemoryBackend.
    Can switch to RedisBackend via cache_type='redis'.
    """

    def __init__(
        self, 
        max_size: int = 10000, 
        ttl_seconds: int = 3600,
        cache_type: str = "memory",
        redis_url: str = None
    ):
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()
        
        if cache_type == "redis":
            if not redis_url:
                raise ValueError("redis_url required for cache_type='redis'")
            self.backend = RedisBackend(redis_url)
        else:
            self.backend = MemoryBackend(max_size)

    @staticmethod
    def make_key(features: Dict[str, Any]) -> str:
        """Generate deterministic cache key from feature dict."""
        sorted_items = sorted(features.items())
        canonical = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict]:
        try:
            val = self.backend.get(key)
            if val:
                self.stats.hits += 1
                return val
            self.stats.misses += 1
            return None
        except Exception:
            self.stats.errors += 1
            self.stats.misses += 1
            return None

    def put(self, key: str, value: Dict) -> None:
        try:
            self.backend.put(key, value, self.ttl_seconds)
        except Exception:
            self.stats.errors += 1

    def invalidate(self) -> None:
        self.backend.clear()
        self.stats = CacheStats()
    
    @property
    def size(self) -> int:
        return self.backend.size()
