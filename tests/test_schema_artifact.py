import json
from pathlib import Path

from bixolon_scanner.contracts import ScanResponse


def test_versioned_response_schema_matches_public_model_fields():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "scan-response.schema.json").read_text())
    assert set(schema["required"]) == set(ScanResponse.model_fields)
    assert schema["properties"]["status"]["enum"] == [
        "SEGMENTATION",
        "IMAGE_RECAPTURE",
        "ERROR",
    ]
    assert schema["$defs"]["Segmentation"]["properties"]["status"]["enum"] == [
        "APPROVED",
        "UNKNOWN",
        "SEGMENT_RECAPTURE",
    ]
