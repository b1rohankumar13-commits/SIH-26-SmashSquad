"""Event-category and official-report context for the tabular branch."""

def build_event_features(event: dict) -> dict:
    return {key: event.get(key) for key in ("lead_day", "region_id", "category_tags")}
