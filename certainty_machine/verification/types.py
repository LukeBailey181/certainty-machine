from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VerificationOutput:
    verdict: bool
    output: Dict[str, Any]
