"""Small note helpers. No database or network required."""

from dataclasses import dataclass


@dataclass
class Note:
    title: str
    content: str
    tags: tuple[str, ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.content.split())
