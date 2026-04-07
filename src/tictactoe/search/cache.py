from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class BoundedCache(Generic[K, V]):
    max_size: int = 4096
    _items: OrderedDict[K, V] = field(default_factory=OrderedDict)

    def get(self, key: K) -> V | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        if len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
