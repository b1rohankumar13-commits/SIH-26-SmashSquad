"""Combine category evidence into one overall historical bust label."""

def overall_bust_label(category_results: list[dict]) -> int:
    return int(any(result.get("bust", False) for result in category_results))
