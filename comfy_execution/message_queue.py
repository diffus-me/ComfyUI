import asyncio

from protocol import BinaryEventTypes


MESSAGE_BACKLOG_COMPACT_THRESHOLD = 8


def _message_compaction_key(message):
    event, data, sid = message

    if event == "status":
        return event, sid

    if event == "progress" and isinstance(data, dict):
        prompt_id = data.get("prompt_id")
        if prompt_id:
            return event, sid, prompt_id, data.get("node")

    if event == "progress_state" and isinstance(data, dict):
        prompt_id = data.get("prompt_id")
        if prompt_id:
            return event, sid, prompt_id

    if event == "preview" and isinstance(data, dict):
        prompt_id = data.get("prompt_id")
        if prompt_id:
            return event, sid, prompt_id, data.get("node")

    if event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            metadata = data[1]
            prompt_id = metadata.get("prompt_id")
            if prompt_id:
                return event, sid, prompt_id, metadata.get("node_id")

    if event in (BinaryEventTypes.PREVIEW_IMAGE, BinaryEventTypes.UNENCODED_PREVIEW_IMAGE):
        return event, sid

    return None


def compact_messages(messages):
    compacted = []
    pending = []
    dropped = 0

    def flush_pending():
        nonlocal dropped
        seen = set()
        latest = []
        for key, message in reversed(pending):
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            latest.append(message)
        compacted.extend(reversed(latest))
        pending.clear()

    for message in messages:
        key = _message_compaction_key(message)
        if key is None:
            flush_pending()
            compacted.append(message)
        else:
            pending.append((key, message))
    flush_pending()

    return compacted, dropped


async def get_compacted_messages(message_queue, threshold=MESSAGE_BACKLOG_COMPACT_THRESHOLD):
    first = await message_queue.get()
    if message_queue.qsize() + 1 < threshold:
        return [first], 0

    messages = [first]
    while True:
        try:
            messages.append(message_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    return compact_messages(messages)
