
"""Deterministic data-quality incident detection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "customer_id",
    "product_id",
    "quantity",
    "unit_price",
    "total_amount",
}

REQUIRED_VALUES = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "product_id",
    "quantity",
    "unit_price",
    "total_amount",
]


def _incident(
    incident_type: str,
    severity: str,
    dataset: str,
    column: str | None,
    explanation: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistently structured incident record."""
    return {
        "incident_type": incident_type,
        "severity": severity,
        "dataset": dataset,
        "column": column,
        "explanation": explanation,
        "evidence": evidence,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def detect_incidents(
    data: pd.DataFrame,
    dataset_name: str = "sales",
    baseline_count: int | None = None,
    volume_drop_threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Detect common data-quality issues without modifying the input."""
    if baseline_count is not None and baseline_count < 0:
        raise ValueError("baseline_count cannot be negative")

    if not 0 <= volume_drop_threshold < 1:
        raise ValueError("volume_drop_threshold must be between 0 and 1")

    incidents: list[dict[str, Any]] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))

    if missing_columns:
        incidents.append(
            _incident(
                incident_type="schema_drift",
                severity="critical",
                dataset=dataset_name,
                column=None,
                explanation="Required columns are missing from the dataset.",
                evidence={"missing_columns": missing_columns},
            )
        )
        return incidents

    for column in REQUIRED_VALUES:
        blank_mask = data[column].map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        missing_mask = data[column].isna() | blank_mask

        if missing_mask.any():
            incidents.append(
                _incident(
                    incident_type="missing_values",
                    severity="high",
                    dataset=dataset_name,
                    column=column,
                    explanation=f"Column '{column}' contains missing values.",
                    evidence={
                        "missing_count": int(missing_mask.sum()),
                        "affected_rows": [
                            str(index) for index in data.index[missing_mask][:10]
                        ],
                    },
                )
            )

    duplicate_mask = data["transaction_id"].duplicated(keep=False)

    if duplicate_mask.any():
        incidents.append(
            _incident(
                incident_type="duplicate_transactions",
                severity="high",
                dataset=dataset_name,
                column="transaction_id",
                explanation="Transaction IDs are not unique.",
                evidence={
                    "duplicate_row_count": int(duplicate_mask.sum()),
                    "duplicate_ids": (
                        data.loc[duplicate_mask, "transaction_id"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()[:10]
                    ),
                },
            )
        )

    numeric_columns = ["quantity", "unit_price", "total_amount"]
    numeric_values: dict[str, pd.Series] = {}

    for column in numeric_columns:
        # Keep missing and blank values in the missing-value category, rather
        # than reporting them as failed numeric conversions.
        source = data[column]
        numeric = pd.to_numeric(source, errors="coerce")
        non_empty_mask = source.notna() & source.astype(str).str.strip().ne("")
        invalid_numeric_mask = non_empty_mask & numeric.isna()
        numeric_values[column] = numeric

        if invalid_numeric_mask.any():
            incidents.append(
                _incident(
                    incident_type="invalid_numeric_value",
                    severity="high",
                    dataset=dataset_name,
                    column=column,
                    explanation=f"Column '{column}' contains non-empty values that cannot be parsed as numbers.",
                    evidence={
                        "column": column,
                        "invalid_count": int(invalid_numeric_mask.sum()),
                        "affected_rows": [
                            str(index)
                            for index in data.index[invalid_numeric_mask][:10]
                        ],
                    },
                )
            )

        invalid_amount_mask = numeric.notna() & (numeric <= 0)
        if invalid_amount_mask.any():
            incidents.append(
                _incident(
                    incident_type="invalid_amounts",
                    severity="high",
                    dataset=dataset_name,
                    column=column,
                    explanation=f"Column '{column}' contains non-positive values.",
                    evidence={
                        "invalid_count": int(invalid_amount_mask.sum()),
                        "affected_rows": [
                            str(index)
                            for index in data.index[invalid_amount_mask][:10]
                        ],
                    },
                )
            )

    # Compare currency values at cent precision. Rounding the calculated total
    # avoids binary floating-point artifacts while still detecting cent-level
    # inconsistencies in the recorded amount.
    valid_arithmetic_mask = pd.Series(True, index=data.index)
    for column in numeric_columns:
        valid_arithmetic_mask &= numeric_values[column].notna()

    if valid_arithmetic_mask.any():
        expected_totals = (
            numeric_values["quantity"] * numeric_values["unit_price"]
        ).round(2)
        actual_totals = numeric_values["total_amount"]
        inconsistent_mask = valid_arithmetic_mask & (
            (expected_totals - actual_totals).abs() > 1e-9
        )

        if inconsistent_mask.any():
            affected_indices = data.index[inconsistent_mask][:10]
            incidents.append(
                _incident(
                    incident_type="inconsistent_total_amount",
                    severity="high",
                    dataset=dataset_name,
                    column="total_amount",
                    explanation="Total amount does not equal quantity multiplied by unit price.",
                    evidence={
                        "affected_rows": [str(index) for index in affected_indices],
                        "expected_totals": {
                            str(index): float(expected_totals.loc[index])
                            for index in affected_indices
                        },
                        "actual_totals": {
                            str(index): float(actual_totals.loc[index])
                            for index in affected_indices
                        },
                    },
                )
            )

    if baseline_count is not None:
        current_count = len(data)

        if baseline_count > 0:
            drop_fraction = (baseline_count - current_count) / baseline_count

            if drop_fraction >= volume_drop_threshold:
                incidents.append(
                    _incident(
                        incident_type="volume_drop",
                        severity="high",
                        dataset=dataset_name,
                        column=None,
                        explanation="Transaction volume dropped beyond the configured threshold.",
                        evidence={
                            "baseline_count": baseline_count,
                            "current_count": current_count,
                            "drop_fraction": round(drop_fraction, 4),
                            "threshold": volume_drop_threshold,
                        },
                    )
                )

    return incidents
