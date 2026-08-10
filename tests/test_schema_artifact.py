import json
from pathlib import Path

from bixolon_scanner.contracts import ScanResponse


def test_versioned_response_schema_matches_public_model_fields():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas" / "scan-response.schema.json").read_text())
    assert set(schema["required"]) == set(ScanResponse.model_fields)
    assert schema["properties"]["status"]["enum"] == [
        "APPROVED",
        "UNKNOWN",
        "RECAPTURE",
        "ERROR",
    ]
    assert schema["$defs"]["ScanItem"]["properties"]["status"]["enum"] == [
        "APPROVED",
        "UNKNOWN",
    ]
