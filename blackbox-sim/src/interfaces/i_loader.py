from abc import ABC, abstractmethod
from src.common.types import LogSample
from typing import Iterator

class ILoader(ABC):
    @abstractmethod
    def load_source(self, path: str) -> None:
        """Opens the file source."""
        pass

    @abstractmethod
    def next_batch(self, size: int) -> Iterator[LogSample]:
        """Returns the next batch of data."""
        pass