"""Simple DTO for token data."""
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ChainDTO:
    """
    Data Transfer Object to store relevant
    chain data.
    """

    name: str
    network_id: int

    def __hash__(self):
        return hash(f"{self.name}-{self.network_id}")