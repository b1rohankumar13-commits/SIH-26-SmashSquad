"""Attach one or more context categories to a detected bust event."""

CATEGORIES = (
    "heavy_rainfall", "monsoon_depression", "cyclone", "heatwave",
    "western_disturbance", "monsoon_phase",
)

def classify_categories(event: dict) -> list[str]:
    raise NotImplementedError("Implement category rules after pilot-data inspection.")
