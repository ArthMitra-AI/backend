_memory_cache: dict = {}

class UpstashCache:
    def is_available(self): return False
    async def get(self, key):
        return _memory_cache.get(key)
    async def set(self, key, value, ex=3600):
        _memory_cache[key] = value
    async def delete(self, key):
        _memory_cache.pop(key, None)
    async def incr(self, key, ex=3600):
        current = int(_memory_cache.get(key, 0))
        _memory_cache[key] = current + 1
        return current + 1

_instance = None

def get_cache():
    global _instance
    if _instance is None:
        _instance = UpstashCache()
    return _instance

async def rate_limit_check(user_id, action, limit, window_seconds=3600):
    return True