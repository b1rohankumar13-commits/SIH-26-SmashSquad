from src.detection.overall_bust_label import overall_bust_label

def test_any_category_can_trigger_overall_label():
    assert overall_bust_label([{"bust": False}, {"bust": True}]) == 1
