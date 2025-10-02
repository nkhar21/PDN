import numpy as np
from typing import List, Tuple, Optional
from generator.gen_outline import OutlineGeneratorFactory, OutlineMode
from generator.gen_segments import generate_segments


class BoardOutline:
    """
    Encapsulates board outline geometry and segmentation data.
    """

    def __init__(self):
        # Outline polygons (N_layers,), each entry is (M,2) ndarray
        self.bxy: Optional[np.ndarray] = None

        # Segmentation results
        self.sxy: Optional[np.ndarray] = None                  # concatenated [x1,y1,x2,y2] per segment
        self.seg_len: Optional[float] = None                   # segment length used
        self.sxy_index_ranges: Optional[List[Tuple[int,int]]] = None
        self.sxy_list: Optional[List[np.ndarray]] = None

    def set_outline(self, shapes: List[np.ndarray], mode: OutlineMode, units: str = "mm"):
        """
        Assign the outline polygons.
        """
        if not shapes:
            raise ValueError("At least one outline polygon must be provided")
        gen = OutlineGeneratorFactory.create(mode, units=units)
        self.bxy = gen.generate(shapes)

    def set_segmentation(self, seg_len: float):
        """
        Segment the outline with a given segment length.
        """
        if self.bxy is None:
            raise ValueError("bxy must be set before segmentation")
        if seg_len <= 0:
            raise ValueError("seg_len must be > 0")

        self.seg_len = seg_len
        sxy, sxy_index_ranges, sxy_list = generate_segments(self.bxy, seg_len)
        self.sxy = sxy
        self.sxy_index_ranges = sxy_index_ranges
        self.sxy_list = sxy_list

    def summary(self):
        """
        Print debug info about the outline.
        """
        print("=== BoardOutline Summary ===")
        if self.bxy is not None:
            print(f"Polygons: {len(self.bxy)}")
        if self.sxy is not None:
            print(f"Segments: {self.sxy.shape}")
        print("============================")
