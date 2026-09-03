import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pydicom

@dataclass
class ImageSchema:
    expected_modality: str
    min_dimensions: Tuple[int, ...]
    allowed_photometric: List[str]
    require_mask: bool = True

class MedicalDataGuardrail:
    """Validates local medical imaging datasets against expected schema before training."""
    
    def __init__(self, schema: ImageSchema):
        self.schema = schema

    def validate_dicom_file(self, image_path: str, mask_path: Optional[str] = None) -> Dict[str, any]:
        errors = []
        
        # 1. File existence & basic header readability
        if not os.path.exists(image_path):
            return {"valid": False, "errors": ["Image file does not exist."]}
            
        try:
            ds = pydicom.dcmread(image_path, stop_before_pixels=True)
        except Exception as e:
            return {"valid": False, "errors": [f"Corrupted header: {str(e)}"]}

        # 2. Modality Verification
        modality = getattr(ds, "Modality", None)
        if modality != self.schema.expected_modality:
            errors.append(f"Invalid Modality: Expected {self.schema.expected_modality}, got {modality}")

        # 3. Spatial Dimension Checks
        rows = getattr(ds, "Rows", 0)
        cols = getattr(ds, "Columns", 0)
        if rows < self.schema.min_dimensions[0] or cols < self.schema.min_dimensions[1]:
            errors.append(f"Dimension mismatch: Got ({rows}, {cols}), required min {self.schema.min_dimensions}")

        # 4. Photometric Interpretation (Color space / Inversion check)
        photo_interp = getattr(ds, "PhotometricInterpretation", "")
        if photo_interp not in self.schema.allowed_photometric:
            errors.append(f"Unsupported Photometric Interpretation: {photo_interp}")

        # 5. Annotation / Mask Existence Check
        if self.schema.require_mask:
            if not mask_path or not os.path.exists(mask_path):
                errors.append("Target annotation/mask file missing.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "metadata": {
                "patient_id": getattr(ds, "PatientID", "UNKNOWN"),
                "dimensions": (rows, cols),
                "modality": modality
            }
        }