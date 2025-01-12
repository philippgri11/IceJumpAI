from dataclasses import dataclass

@dataclass
class AI:
    name = "Easy"

    vecX: float = 0

    def __init__(self, name: str):
        self.name = name

    def think(self, index: int):
        pass