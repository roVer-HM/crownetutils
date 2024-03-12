from enum import Enum


class MapType(Enum):
    DENSITY = "density"
    ENTROPY = "entropy"

    def __str__(self) -> str:
        return self.value
