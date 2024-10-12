import logging
import diffus.redis_client

logger = logging.getLogger(__name__)


class MessageQueue:
    def __init__(self):
        self._redis_client = diffus.redis_client.get_redis_client()

    def send_message(self, sid: str, message: bytes | str, retry=1):
        if not self._redis_client:
            self._redis_client = diffus.redis_client.get_redis_client()
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
