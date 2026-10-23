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


def test_arithmetic_evidence_handles_duplicate_index_labels():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 5
    data.loc[0, "unit_price"] = 100.00
    data.loc[0, "total_amount"] = 700.00
    data.loc[1, "quantity"] = 2
    data.loc[1, "unit_price"] = 20.00
    data.loc[1, "total_amount"] = 100.00
    data.index = [0, 0, *range(2, 10)]
    original = data.copy(deep=True)

    incidents = detect_incidents(data)
    incident = next(
        item for item in incidents
        if item["incident_type"] == "inconsistent_total_amount"
    )
    evidence = incident["evidence"]

    assert evidence["affected_count"] == 2
    assert evidence["affected_rows"] == ["0 (position 0)", "0 (position 1)"]
    assert evidence["expected_totals"] == {
        "0 (position 0)": 500.0,
        "0 (position 1)": 40.0,
    }
    assert evidence["actual_totals"] == {
        "0 (position 0)": 700.0,
        "0 (position 1)": 100.0,
    }
    pd.testing.assert_frame_equal(data, original)


def test_arithmetic_evidence_count_includes_rows_beyond_evidence_limit():
    data = generate_sales_data(rows=12)
    data["quantity"] = 2
    data["unit_price"] = 10.00
    data["total_amount"] = 999.00

    incidents = detect_incidents(data)
    incident = next(
        item for item in incidents
        if item["incident_type"] == "inconsistent_total_amount"
    )

    assert incident["evidence"]["affected_count"] == 12
    assert len(incident["evidence"]["affected_rows"]) == 10


@pytest.mark.parametrize(
    "baseline_count",
    [True, False, "10", 10.5, float("nan"), float("inf"), float("-inf"), -1],
)
def test_invalid_baseline_count_is_rejected(baseline_count):
    data = generate_sales_data(rows=10)

    with pytest.raises(ValueError):
        detect_incidents(data, baseline_count=baseline_count)


def test_duplicate_required_column_names_return_schema_drift():
    data = generate_sales_data(rows=10)
    data.columns = [*data.columns[:-1], "quantity"]

    incidents = detect_incidents(data)

    schema_incident = next(
        item for item in incidents if item["incident_type"] == "schema_drift"
    )
    assert schema_incident["evidence"]["duplicate_columns"] == ["quantity"]


def test_duplicate_column_names_and_missing_required_columns_are_reported_together():
    data = generate_sales_data(rows=10).rename(columns={"customer_id": "client_id"})
    data.columns = [*data.columns[:-1], "quantity"]

    incidents = detect_incidents(data)

    schema_incident = next(
        item for item in incidents if item["incident_type"] == "schema_drift"
    )
    assert schema_incident["evidence"]["duplicate_columns"] == ["quantity"]
    assert schema_incident["evidence"]["missing_columns"] == ["customer_id", "total_amount"]


@pytest.mark.parametrize(
    ("column", "value", "incident_type", "count_key"),
    [
        ("customer_id", None, "missing_values", "missing_count"),
        ("quantity", "unknown", "invalid_numeric_value", "invalid_count"),
        ("quantity", 0, "invalid_amounts", "invalid_count"),
    ],
)
def test_row_level_evidence_disambiguates_duplicate_index_labels(
    column, value, incident_type, count_key
):
    data = generate_sales_data(rows=10)
    data[column] = data[column].astype(object)
    data.iloc[0, data.columns.get_loc(column)] = value
    data.iloc[1, data.columns.get_loc(column)] = value
    data.index = [7, 7, *range(2, 10)]

    incidents = detect_incidents(data)
    incident = next(
        item for item in incidents
        if item["incident_type"] == incident_type and item["column"] == column
    )

    assert incident["evidence"][count_key] == 2
    assert incident["evidence"]["affected_rows"] == [
        "7 (position 0)", "7 (position 1)"
    ]


def test_cent_precision_ignores_sub_cent_difference():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 1
    data.loc[0, "unit_price"] = 1.00
    data.loc[0, "total_amount"] = 1.004

    incidents = detect_incidents(data)

    assert not any(
        item["incident_type"] == "inconsistent_total_amount" for item in incidents
    )


def test_cent_precision_detects_one_cent_difference():
    data = generate_sales_data(rows=10)
    data.loc[0, "quantity"] = 1
    data.loc[0, "unit_price"] = 1.00
    data.loc[0, "total_amount"] = 1.01

    incidents = detect_incidents(data)

    incident = next(
        item for item in incidents
        if item["incident_type"] == "inconsistent_total_amount"
    )
    assert incident["evidence"]["expected_totals"]["0"] == 1.0
    assert incident["evidence"]["actual_totals"]["0"] == 1.01
