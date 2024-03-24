import json

import redis
from loguru import logger


class RedisCache:
    def __init__(self, host="localhost", port=6379):
        logger.info("Connecting to Redis at {}:{}", host, port)
        self.r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        logger.info("Connected to Redis at {}:{}", host, port)

    def set(self, key, value, ttl=None):
        self.r.set(key, json.dumps(value))
        if ttl:
            self.r.expire(key, ttl)

    def get(self, key):
        result = self.r.get(key)
        return json.loads(result) if result else None
