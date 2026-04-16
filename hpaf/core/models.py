from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Pose:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

@dataclass
class Detection:
    label: str
    bbox: List[int]
    score: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
