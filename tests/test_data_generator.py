
import pandas as pd
import pytest

from src.data_generator import generate_sales_data, inject_incident


def test_generator_is_reproducible():
    first = generate_sales_data(rows=20, seed=42)
    second = generate_sales_data(rows=20, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_generator_creates_expected_columns():
    data = generate_sales_data(rows=20)

    expected = {
        "transaction_id",
        "timestamp",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "total_amount",
    }

    assert set(data.columns) == expected
    assert len(data) == 20


@pytest.mark.parametrize(
    "incident",
    [
        "missing_values",
        "duplicate_transactions",
        "invalid_amounts",
        "schema_drift",
    ],
)
def test_incident_injection_changes_a_copy(incident):
    original = generate_sales_data(rows=20)
    original_copy = original.copy(deep=True)

    modified = inject_incident(original, incident)

    pd.testing.assert_frame_equal(original, original_copy)
    assert not modified.equals(original)


def test_invalid_row_count_is_rejected():
    with pytest.raises(ValueError):
        generate_sales_data(rows=0)
