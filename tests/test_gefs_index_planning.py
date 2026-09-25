from src.acquisition.download_gefs_reforecast import GribRecord, _matches, parse_index


def test_parse_index_infers_inclusive_ranges():
    records = parse_index("1:0:d=x:TMP:2 m above ground:fcst\n2:100:d=x:APCP:surface:acc", 250)
    assert records == [
        GribRecord(1, 0, 99, "d=x:TMP:2 m above ground:fcst"),
        GribRecord(2, 100, 249, "d=x:APCP:surface:acc"),
    ]


def test_selector_matches_variable_and_level_exactly():
    record = GribRecord(1, 0, 99, "d=x:TMP:850 mb:24 hour fcst")
    assert _matches(record, {"variable": "TMP", "level": "850 mb"})
    assert not _matches(record, {"variable": "TMP", "level": "500 mb"})
