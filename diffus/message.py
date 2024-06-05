import logging
import os
import time

import redis

logger = logging.getLogger(__name__)


def get_redis_client() -> redis.Redis:
    redis_address = os.getenv('REDIS_ADDRESS')
    while True:
        try:
            if redis_address:
                redis_client = redis.Redis.from_url(url=redis_address)
                redis_client.ping()
            else:
                redis_client = None
            return redis_client
        except Exception as e:
            logger.exception(f'failed to create redis client from {redis_address}: {e.__str__()}')
            time.sleep(3)


class MessageQueue:
    def __init__(self):
        self._redis_client = get_redis_client()

    def send_message(self, sid: str, message: bytes | str, retry=1):
        if not self._redis_client:
            self._redis_client = get_redis_client()
        try:
            if sid:
                self._redis_client.rpush(f'COMFYUI_MESSAGE_{sid}', message)
            else:
                self._redis_client.publish(f'COMFYUI_MESSAGE_anonymous', message)
        except Exception as e:
            logger.exception(e)
            if retry > 0:
                self._redis_client = None
                self.send_message(sid, message, retry=retry - 1)
