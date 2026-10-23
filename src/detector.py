
"""Deterministic data-quality incident detection."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from numbers import Integral
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
    if baseline_count is not None and (
        isinstance(baseline_count, bool)
        or not isinstance(baseline_count, Integral)
        or baseline_count < 0
    ):
        raise ValueError("baseline_count must be a non-negative integer")

    if not 0 <= volume_drop_threshold < 1:
        raise ValueError("volume_drop_threshold must be between 0 and 1")

    incidents: list[dict[str, Any]] = []

    duplicate_columns = sorted(
        str(column) for column in data.columns[data.columns.duplicated(keep=False)].unique()
    )
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if duplicate_columns:
        schema_evidence: dict[str, Any] = {"duplicate_columns": duplicate_columns}
        if missing_columns:
            schema_evidence["missing_columns"] = missing_columns
        incidents.append(
            _incident(
                incident_type="schema_drift",
                severity="critical",
                dataset=dataset_name,
                column=None,
                explanation="The dataset contains duplicate column names.",
                evidence=schema_evidence,
            )
        )
        return incidents


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

    def row_reference(position: int) -> str:
        label = str(data.index[position])
        if data.index.is_unique:
            return label
        return f"{label} (position {position})"

    def affected_row_references(mask: pd.Series, limit: int = 10) -> list[str]:
        return [
            row_reference(position)
            for position, is_affected in enumerate(mask.to_numpy(dtype=bool))
            if is_affected
        ][:limit]

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
                        "affected_rows": affected_row_references(missing_mask),
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
        non_finite_mask = numeric.isin([float("inf"), float("-inf")])
        invalid_numeric_mask = non_empty_mask & (numeric.isna() | non_finite_mask)
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
                        "affected_rows": affected_row_references(invalid_numeric_mask),
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
                        "affected_rows": affected_row_references(invalid_amount_mask),
                    },
                )
            )

    # Compare currency values at cent precision using row positions. Index labels
    # are not guaranteed to be unique, so they must not be used for matching.
    valid_positions: list[int] = []
    expected_by_position: dict[int, float] = {}
    actual_by_position: dict[int, float] = {}

    for position in range(len(data)):
        values = {
            column: numeric_values[column].iloc[position]
            for column in numeric_columns
        }
        if not all(
            pd.notna(value) and value not in (float("inf"), float("-inf"))
            for value in values.values()
        ):
            continue

        valid_positions.append(position)
        expected_by_position[position] = float(
            (Decimal(str(values["quantity"])) * Decimal(str(values["unit_price"])))
            .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
        actual_by_position[position] = float(
            Decimal(str(values["total_amount"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )

    inconsistent_positions = [
        position
        for position in valid_positions
        if expected_by_position[position] != actual_by_position[position]
    ]

    if inconsistent_positions:

        evidence_positions = inconsistent_positions[:10]
        row_references = {
            position: row_reference(position) for position in evidence_positions
        }

        incidents.append(
            _incident(
                incident_type="inconsistent_total_amount",
                severity="high",
                dataset=dataset_name,
                column="total_amount",
                explanation="Total amount does not equal quantity multiplied by unit price.",
                evidence={
                    "affected_count": len(inconsistent_positions),
                    "affected_rows": [
                        row_references[position] for position in evidence_positions
                    ],
                    "expected_totals": {
                        row_references[position]: expected_by_position[position]
                        for position in evidence_positions
                    },
                    "actual_totals": {
                        row_references[position]: actual_by_position[position]
                        for position in evidence_positions
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
