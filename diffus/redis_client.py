import os

import redis


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
