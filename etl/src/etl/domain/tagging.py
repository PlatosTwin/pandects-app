from dataclasses import dataclass
from typing import List


@dataclass
class TagData:
    tag: str


def tag(rows, classifier_model) -> List[TagData]:
    pass
