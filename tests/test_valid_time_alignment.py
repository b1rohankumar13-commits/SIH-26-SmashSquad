from datetime import datetime
from src.utils.time_utils import valid_time

def test_valid_time():
    assert valid_time(datetime(2026, 1, 1), 240) == datetime(2026, 1, 11)
