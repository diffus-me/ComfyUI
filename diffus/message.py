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
                keys = [
                    f'COMFYUI_MESSAGE_{sid}',
                    f'diffus:comfyui:message:{sid}'
                ]
            else:
                keys = [
                    f'COMFYUI_MESSAGE_anonymous',
                    f'diffus:comfyui:message:anonymous'
                ]
            for key in keys:
                self._redis_client.rpush(key, message)
                self._redis_client.expire(key, 60 * 60)
        except Exception as e:
            logger.exception(e)
            if retry > 0:
                self._redis_client = None
                self.send_message(sid, message, retry=retry - 1)
