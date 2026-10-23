import pandas as pd
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

def test_invalid_numeric_evidence_identifies_column_and_rows():
    data = generate_sales_data(rows=10)
    data["quantity"] = data["quantity"].astype(object)
    data.loc[0, "quantity"] = "unknown"

    incidents = detect_incidents(data)
    incident = next(
        item for item in incidents if item["incident_type"] == "invalid_numeric_value"
    )

    assert incident["column"] == "quantity"
    assert incident["evidence"]["column"] == "quantity"
    assert incident["evidence"]["affected_rows"] == ["0"]


def test_missing_numeric_value_is_not_reported_as_invalid_numeric():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = float("nan")

    incidents = detect_incidents(data)

    assert any(
        item["incident_type"] == "missing_values" and item["column"] == "quantity"
        for item in incidents
    )
    assert not any(item["incident_type"] == "invalid_numeric_value" for item in incidents)


def test_arithmetic_incident_includes_expected_and_actual_totals():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 5
    data.loc[0, "unit_price"] = 100.00
    data.loc[0, "total_amount"] = 700.00

    incidents = detect_incidents(data)
    incident = next(
        item for item in incidents if item["incident_type"] == "inconsistent_total_amount"
    )

    assert incident["evidence"]["affected_rows"] == ["0"]
    assert incident["evidence"]["expected_totals"]["0"] == 500.0
    assert incident["evidence"]["actual_totals"]["0"] == 700.0


def test_floating_point_rounding_does_not_create_false_inconsistency():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 3
    data.loc[0, "unit_price"] = 0.1
    data.loc[0, "total_amount"] = 0.30

    incidents = detect_incidents(data)

    assert not any(
        item["incident_type"] == "inconsistent_total_amount" for item in incidents
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.0, 2.0])
def test_invalid_volume_threshold_is_rejected_without_baseline(threshold):
    data = generate_sales_data(rows=10)

    with pytest.raises(ValueError):
        detect_incidents(data, volume_drop_threshold=threshold)


@pytest.mark.parametrize("column", ["quantity", "unit_price", "total_amount"])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_blank_numeric_values_are_reported_as_missing_not_invalid_numeric(
    column, blank_value
):
    data = generate_sales_data(rows=10)
    data[column] = data[column].astype(object)
    data.loc[0, column] = blank_value
    original = data.copy(deep=True)

    incidents = detect_incidents(data)

    assert any(
        item["incident_type"] == "missing_values" and item["column"] == column
        and item["evidence"]["affected_rows"] == ["0"]
        for item in incidents
    )
    assert not any(
        item["incident_type"] == "invalid_numeric_value"
        and item["column"] == column
        for item in incidents
    )
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("column", ["quantity", "unit_price", "total_amount"])
@pytest.mark.parametrize("non_finite_value", [float("inf"), float("-inf")])
def test_non_finite_numeric_values_are_reported_as_invalid_numeric(
    column, non_finite_value
):
    data = generate_sales_data(rows=10)
    data[column] = data[column].astype(object)
    data.loc[0, column] = non_finite_value
    original = data.copy(deep=True)

    incidents = detect_incidents(data)

    assert any(
        item["incident_type"] == "invalid_numeric_value"
        and item["column"] == column
        and item["evidence"]["affected_rows"] == ["0"]
        for item in incidents
    )
    pd.testing.assert_frame_equal(data, original)


def test_non_finite_values_are_excluded_from_arithmetic_consistency_check():
    data = generate_sales_data(rows=10)
    data["quantity"] = data["quantity"].astype(object)
    data.loc[0, "quantity"] = float("inf")

    incidents = detect_incidents(data)

    assert any(
        item["incident_type"] == "invalid_numeric_value"
        and item["column"] == "quantity"
        for item in incidents
    )
    assert not any(
        item["incident_type"] == "inconsistent_total_amount"
        for item in incidents
    )
