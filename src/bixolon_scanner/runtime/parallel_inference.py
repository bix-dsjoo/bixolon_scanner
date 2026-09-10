"""Join independent CPU/GPU evidence before any decision or resource recovery."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


def verification_pair(rotation, independent, *, parallel: bool):
    if not parallel:
        return rotation(), independent()
    # Joining on every exit also prevents CPU inference racing provider recovery.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifier") as executor:
        future = executor.submit(copy_context().run, independent)
        rotated = rotation()
        return rotated, future.result()
