#!/usr/bin/env python3
import sys
from gc import get_referents
from types import FunctionType
from types import ModuleType

import torch

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj: object):
    """sum size of object & members."""

    if isinstance(obj, BLACKLIST):
        raise TypeError(
            "getsize() does not take argument of type: " + str(type(obj))
        )

    seen_ids = set()
    size = 0
    objects = [obj]

    count = 0

    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                if isinstance(obj, torch.Tensor):
                    obj = obj.storage()
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
                count += 1
        objects = get_referents(*need_referents)

    size = f"{count} objects, size of " + human_readable_size(size)

    return size


def human_readable_size(size, decimal_places=3):
    """Convert size to human readable unit."""

    final_unit = ""

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:

        if size < 1024.0:
            final_unit = unit
            break
        size /= 1024.0

    return f"{size:.{decimal_places}f} {final_unit}"
