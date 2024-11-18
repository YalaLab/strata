# This is to prevent unsloth from mis-recognizing our setting to be multi-gpu setting if we run multiple training processes in parallel.

from functools import lru_cache
from unsloth import tokenizer_utils
from unsloth.models import _utils
from unsloth.models import llama
import subprocess

original_check_output = subprocess.check_output
cached_check_output = lru_cache(maxsize=1)(original_check_output)


def patched_check_output(*args, **kwargs):
    if args == ("nvidia-smi --query-gpu=memory.used --format=csv",) and kwargs == {
        "shell": True
    }:
        return cached_check_output(*args, **kwargs)
    return original_check_output(*args, **kwargs)


def patch_unsloth():
    subprocess.check_output = patched_check_output
    tokenizer_utils.check_nvidia = lru_cache(maxsize=1)(tokenizer_utils.check_nvidia)
    _utils.check_nvidia = lru_cache(maxsize=1)(_utils.check_nvidia)
    llama.check_nvidia = lru_cache(maxsize=1)(llama.check_nvidia)
