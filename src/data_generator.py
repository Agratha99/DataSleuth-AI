
"""Generate reproducible synthetic sales data for DataSleuth AI."""

from __future__ import annotations

from typing import Literal

import pandas as pd


IncidentType = Literal[
    "missing_values",
    "duplicate_transactions",
    "invalid_amounts",
    "schema_drift",
]


def generate_sales_data(
    rows: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic sales dataset."""
    if rows < 1:
        raise ValueError("rows must be at least 1")

    rng = __import__("numpy").random.default_rng(seed)
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="h",
    )

    quantities = rng.integers(1, 11, size=rows)
    unit_prices = rng.uniform(10.0, 500.0, size=rows).round(2)

    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:06d}" for i in range(1, rows + 1)],
            "timestamp": timestamps,
            "customer_id": [
                f"CUST-{value:04d}"
                for value in rng.integers(1, 101, size=rows)
            ],
            "product_id": [
                f"PROD-{value:03d}"
                for value in rng.integers(1, 21, size=rows)
            ],
            "quantity": quantities,
            "unit_price": unit_prices,
            "total_amount": (quantities * unit_prices).round(2),
        }
    )


def inject_incident(
    data: pd.DataFrame,
    incident: IncidentType,
) -> pd.DataFrame:
    """Return a modified copy containing one controlled incident."""
    if data.empty:
        raise ValueError("Cannot inject an incident into an empty dataset")

    result = data.copy(deep=True)

    if incident == "missing_values":
        result.loc[result.index[0], "customer_id"] = None

    elif incident == "duplicate_transactions":
        result.loc[result.index[-1], "transaction_id"] = result.loc[
            result.index[0], "transaction_id"
        ]

    elif incident == "invalid_amounts":
        result.loc[result.index[0], "quantity"] = -5

    elif incident == "schema_drift":
        result = result.rename(columns={"customer_id": "client_id"})

    else:
        raise ValueError(f"Unsupported incident type: {incident}")

    return result
