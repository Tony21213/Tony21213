"""crownai - automatic dental crown design from intraoral scans."""

from .anatomy import ShapeModel, tooth_type_for_fdi
from .design import CrownParameters, CrownResult, design_crown
from .margin import detect_margin
from .mesh import Mesh, load_stl, save_stl

__all__ = ["CrownParameters", "CrownResult", "Mesh", "ShapeModel", "design_crown",
           "detect_margin", "load_stl", "save_stl", "tooth_type_for_fdi"]
__version__ = "0.1.0"
