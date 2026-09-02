"""Evaluate Day 1 through Day 10 separately."""

def evaluate_by_lead(rows):
    """Group event-metric dictionaries by integer lead day."""
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["lead_day"]), []).append(row)
    return dict(sorted(grouped.items()))
