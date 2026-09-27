"""Curated historical case studies (validation-period dates the models never trained on)."""

KERALA_BOX = {"south": 8.2, "north": 12.8, "west": 74.8, "east": 77.5}

CASES = {
    "kerala_2018_d4": {
        "title": "Kerala floods, Aug 2018 · 3 days ahead",
        "hazard": "rain", "init": "20180813", "lead": 4, "mode": "miss",
        "region": "Kerala", "box": KERALA_BOX,
        "story": ("Kerala's worst flood in a century peaked on 15–17 August 2018 after days of extreme "
                  "rain. This forecast was issued on 13 August for the peak day, 16 August."),
    },
    "kerala_2018_d9": {
        "title": "Kerala floods, Aug 2018 · 8 days ahead",
        "hazard": "rain", "init": "20180808", "lead": 9, "mode": "miss",
        "region": "Kerala", "box": KERALA_BOX,
        "story": ("The same peak day (16 August 2018), forecast from 8 August — more than a week before "
                  "the flooding began in earnest."),
    },
    "nw_heat_2009_d9": {
        "title": "North-west India heatwave, Jun 2009 · 8 days ahead",
        "hazard": "heatwave", "init": "20090613", "lead": 9, "mode": "miss",
        "region": "Rajasthan–Haryana", "box": {"south": 24.0, "north": 30.5, "west": 72.0, "east": 79.5},
        "story": ("In June 2009 the monsoon stalled for weeks (2009 became a major drought year) and "
                  "north-west India baked. This forecast was issued on 13 June for 21 June."),
    },
    "central_heat_2009_d4": {
        "title": "Central India heatwave, Jun 2009 · 3 days ahead",
        "hazard": "heatwave", "init": "20090606", "lead": 4, "mode": "miss",
        "region": "central India", "box": {"south": 21.0, "north": 26.5, "west": 78.0, "east": 85.0},
        "story": "Forecast issued on 6 June 2009 for 9 June, as heat built over central India ahead of the delayed monsoon.",
    },
    "central_heat_fa_2009_d4": {
        "title": "Heatwave that never came, Apr 2009 · 3 days ahead",
        "hazard": "heatwave", "init": "20090426", "lead": 4, "mode": "false_alarm",
        "region": "central & west India", "box": {"south": 20.0, "north": 29.0, "west": 72.0, "east": 83.0},
        "story": ("The opposite failure: forecast issued on 26 April 2009 for 29 April. GEFS called a "
                  "widespread heatwave over central and west India — most of it never arrived."),
    },
    # Low-pressure-system leads verify at 00 UTC on init + lead, so lead 7 is 7 days ahead.
    # Storm model validation years are 2016-2019 (trained on 2000-2015).
    "ockhi_2017_d7": {
        "title": "Cyclone Ockhi, Nov 2017 · 7 days ahead",
        "hazard": "lps", "init": "20171123", "lead": 7, "mode": "miss",
        "region": "south-west Bay & Comorin", "box": {"south": 4.5, "north": 10.5, "west": 74.5, "east": 81.0},
        "story": ("Ockhi formed off Sri Lanka and intensified fast as it swept past the southern tip of India "
                  "into the Arabian Sea, catching hundreds of fishing boats at sea. In this forecast, issued "
                  "23 November for 30 November, not one GEFS member had a system there."),
    },
    "fani_2019_d7": {
        "title": "Cyclone Fani, Apr 2019 · 7 days ahead",
        "hazard": "lps", "init": "20190422", "lead": 7, "mode": "miss",
        "region": "south Bay of Bengal", "box": {"south": 5.5, "north": 12.0, "west": 85.0, "east": 90.0},
        "story": ("Fani became an extremely severe cyclone and struck Odisha on 3 May 2019. This forecast "
                  "was issued on 22 April, before Fani had formed, for 29 April, the day our tracker first "
                  "finds it in the south Bay. GEFS had nothing there."),
    },
    "bulbul_2019_d4": {
        "title": "Cyclone Bulbul, Nov 2019 · 4 days ahead",
        "hazard": "lps", "init": "20191103", "lead": 4, "mode": "miss",
        "region": "central Bay of Bengal", "box": {"south": 11.5, "north": 18.0, "west": 86.0, "east": 93.0},
        "story": ("Bulbul tracked north through the Bay to hit West Bengal and Bangladesh on 9 November 2019. "
                  "Issued 3 November for 7 November, GEFS showed no system over the central Bay."),
    },
}
