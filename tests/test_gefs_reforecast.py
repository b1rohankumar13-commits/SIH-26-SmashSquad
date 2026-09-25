import pytest

from src.acquisition.download_gefs_reforecast import (
    add_lengths,
    index_lengths,
    merge_ranges,
    parse_index,
    select_instant,
    select_precip_6h_buckets,
    selected_bytes,
)

INSTANT_IDX = """\
1:0:d=2019070100:UGRD:1000 mb:24 hour fcst:ENS=low-res ctl
2:100:d=2019070100:UGRD:850 mb:24 hour fcst:ENS=low-res ctl
3:250:d=2019070100:UGRD:1000 mb:48 hour fcst:ENS=low-res ctl
4:400:d=2019070100:UGRD:850 mb:48 hour fcst:ENS=low-res ctl
"""

PRECIP_IDX = """\
1:0:d=2019070100:APCP:surface:0-3 hour acc fcst:ENS=low-res ctl
2:200:d=2019070100:APCP:surface:0-6 hour acc fcst:ENS=low-res ctl
3:500:d=2019070100:APCP:surface:6-9 hour acc fcst:ENS=low-res ctl
4:700:d=2019070100:APCP:surface:6-12 hour acc fcst:ENS=low-res ctl
5:1000:d=2019070100:APCP:surface:12-18 hour acc fcst:ENS=low-res ctl
6:1300:d=2019070100:APCP:surface:18-24 hour acc fcst:ENS=low-res ctl
"""


def test_parse_index_reads_fields_and_step_properties():
    records = parse_index(INSTANT_IDX)
    assert len(records) == 4
    assert records[1].variable == "UGRD" and records[1].level == "850 mb"
    assert records[1].instant_hour == 24 and records[1].accumulation is None
    precip = parse_index(PRECIP_IDX)
    assert precip[1].accumulation == (0, 6) and precip[1].instant_hour is None


def test_parse_index_rejects_malformed_and_empty():
    with pytest.raises(ValueError, match="Malformed"):
        parse_index("1:0:d=2019:UGRD\n")
    with pytest.raises(ValueError, match="Empty"):
        parse_index("\n  \n")


def test_add_lengths_uses_next_offset_and_file_size_for_last():
    by_message = {r.message: r for r in add_lengths(parse_index(INSTANT_IDX), file_size=600)}
    assert by_message[1].length == 100
    assert by_message[2].length == 150
    assert by_message[4].length == 200


def test_add_lengths_rejects_non_increasing_offsets():
    bad = "1:0:d=x:UGRD:850 mb:24 hour fcst:ENS=c\n2:0:d=x:UGRD:850 mb:48 hour fcst:ENS=c\n"
    with pytest.raises(ValueError, match="Non-increasing"):
        add_lengths(parse_index(bad), file_size=100)


def test_select_instant_picks_level_and_hours_sorted():
    records = add_lengths(parse_index(INSTANT_IDX), file_size=600)
    chosen = select_instant(records, "UGRD", "850 mb", [24, 48])
    assert [r.message for r in chosen] == [2, 4]
    assert [r.instant_hour for r in chosen] == [24, 48]


def test_select_instant_raises_on_missing_lead_hour():
    records = add_lengths(parse_index(INSTANT_IDX), file_size=600)
    with pytest.raises(ValueError, match="missing lead hours"):
        select_instant(records, "UGRD", "850 mb", [24, 72])


def test_select_precip_keeps_only_six_hour_buckets_to_horizon():
    records = add_lengths(parse_index(PRECIP_IDX), file_size=1600)
    chosen = select_precip_6h_buckets(records, "APCP", lead_days=1)
    assert [r.accumulation for r in chosen] == [(0, 6), (6, 12), (12, 18), (18, 24)]
    assert all(r.accumulation[1] - r.accumulation[0] == 6 for r in chosen)


def test_select_precip_raises_when_a_day_is_incomplete():
    records = add_lengths(parse_index(PRECIP_IDX), file_size=1600)
    with pytest.raises(ValueError, match="missing 6h accumulation"):
        select_precip_6h_buckets(records, "APCP", lead_days=2)


def test_index_lengths_marks_final_message_open_ended():
    by_message = {r.message: r for r in index_lengths(parse_index(INSTANT_IDX))}
    assert by_message[1].length == 100
    assert by_message[2].length == 150
    assert by_message[4].length is None


def test_merge_ranges_open_ended_for_final_message():
    records = index_lengths(parse_index(INSTANT_IDX))
    chosen = [r for r in records if r.message == 4]
    assert merge_ranges(chosen) == [(400, None)]


def test_merge_ranges_collapses_contiguous_messages():
    records = add_lengths(parse_index(PRECIP_IDX), file_size=1600)
    chosen = select_precip_6h_buckets(records, "APCP", lead_days=1)
    assert merge_ranges(chosen) == [(200, 500), (700, 1600)]
    assert selected_bytes(chosen) == (500 - 200) + (1000 - 700) + (1300 - 1000) + (1600 - 1300)
