"""Evaluate reliability and skill by Indian region."""

def evaluate_by_region(rows):
    """Group event-metric dictionaries by region identifier."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["region_id"], []).append(row)
    return grouped
