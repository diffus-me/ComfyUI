import json
import random

from pydantic import BaseModel

import logging

_logger = logging.getLogger(__name__)


class ServiceNode(BaseModel):
    service: str
    status: str = "UP"
    ip: str
    port: str
    schema: str = "http"
    health_check: str = "/health"

    def __key(self):
        return self.service, self.ip, self.port, self.schema

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, ServiceNode):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def alive(self):
        return self.status.lower() == "up"

    @property
    def host_url(self):
        return f"{self.schema}://{self.ip}:{self.port}"

    @staticmethod
    def from_json_str(json_str: str, service_name: str = ''):
        node_dict = json.loads(json_str)
        if service_name:
            node_dict["service"] = service_name
        node_dict["port"] = str(node_dict["port"])
        return ServiceNode(**node_dict)


def get_service_node(redis_client, service_name: str) -> ServiceNode | None:
    service_pattern = f"service:{service_name}_*"
    alive_nodes = []
    for instance_id in redis_client.scan_iter(service_pattern):
        if not instance_id:
            continue
        status_str = redis_client.get(instance_id)
        try:
            node = ServiceNode.from_json_str(status_str, service_name)
            if node.alive:
                alive_nodes.append(node)
        except Exception as e:
            _logger.warning(f"make node status for '{instance_id}' from {status_str} failed: {e.__str__()}")
    if alive_nodes:
        return random.choice(alive_nodes)
    else:
        return None
