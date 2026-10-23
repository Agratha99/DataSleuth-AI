
import pandas as pd
import pytest

from src.data_generator import generate_sales_data, inject_incident
from src.detector import detect_incidents


def test_clean_data_has_no_incidents():
    data = generate_sales_data(rows=100)

    assert detect_incidents(data) == []


@pytest.mark.parametrize(
    ("scenario", "expected_type"),
    [
        ("missing_values", "missing_values"),
        ("duplicate_transactions", "duplicate_transactions"),
        ("invalid_amounts", "invalid_amounts"),
        ("schema_drift", "schema_drift"),
    ],
)
def test_detects_injected_incidents(scenario, expected_type):
    data = generate_sales_data(rows=100)
    faulty_data = inject_incident(data, scenario)

    incidents = detect_incidents(faulty_data)

    assert expected_type in {item["incident_type"] for item in incidents}


def test_detects_volume_drop():
    data = generate_sales_data(rows=60)

    incidents = detect_incidents(
        data,
        baseline_count=100,
        volume_drop_threshold=0.30,
    )

    volume_incidents = [
        item for item in incidents if item["incident_type"] == "volume_drop"
    ]

    assert len(volume_incidents) == 1
    assert volume_incidents[0]["evidence"]["drop_fraction"] == 0.4


def test_detector_does_not_modify_input():
    data = generate_sales_data(rows=20)
    faulty_data = inject_incident(data, "missing_values")
    original = faulty_data.copy(deep=True)

    detect_incidents(faulty_data)

    pd.testing.assert_frame_equal(faulty_data, original)


def test_negative_baseline_is_rejected():
    data = generate_sales_data(rows=10)

    with pytest.raises(ValueError):
        detect_incidents(data, baseline_count=-1)
