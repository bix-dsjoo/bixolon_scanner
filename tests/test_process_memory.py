import os

import pytest

from bixolon_scanner.operations.process_memory import process_memory


@pytest.mark.skipif(os.name != "nt", reason="Windows memory counters")
def test_current_process_and_system_memory_counters():
    result = process_memory(os.getpid())
    assert result["available"]
    assert 0 < result["working_set_bytes"] <= result["peak_working_set_bytes"]
    assert result["private_bytes"] > 0
    assert 0 <= result["system_memory_load_percent"] <= 100
    assert 0 < result["system_available_physical_bytes"] <= result["system_total_physical_bytes"]
