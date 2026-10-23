
import pytest

from src.data_generator import generate_sales_data
from src.detector import detect_incidents


def test_detects_inconsistent_total_amount():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 5
    data.loc[0, "unit_price"] = 100.00
    data.loc[0, "total_amount"] = 700.00

    incidents = detect_incidents(data)

    assert "inconsistent_total_amount" in {
        item["incident_type"] for item in incidents
    }


def test_detects_non_numeric_quantity():
    data = generate_sales_data(rows=10)
    data["quantity"] = data["quantity"].astype(object)
    data.loc[0, "quantity"] = "unknown"

    incidents = detect_incidents(data)

    assert "invalid_numeric_value" in {
        item["incident_type"] for item in incidents
    }


def test_invalid_volume_threshold_is_rejected():
    data = generate_sales_data(rows=10)

    with pytest.raises(ValueError):
        detect_incidents(
            data,
            baseline_count=100,
            volume_drop_threshold=1.0,
        )
