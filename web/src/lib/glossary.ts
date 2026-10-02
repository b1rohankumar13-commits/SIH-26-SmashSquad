/**
 * One-line meanings of the explanation drivers (driver groups in scripts/export_explanations.py
 * and scripts/export_monsoon_explanations.py). Live NEPS runs rename "GEFS" to "NEPS", so lookups
 * normalise that back.
 */
const GLOSSARY: Record<string, string> = {
  // rainfall / storms
  "Forecast rain amount": "Ensemble-mean 24 h rain the forecast predicts here.",
  "Members disagree on rain": "Spread of rain across ensemble members; high spread means an uncertain forecast.",
  "GEFS heavy-rain probability": "Share of members forecasting heavy (≥ 64.5 mm), very heavy or extreme rain.",
  "Heavy rain forecast nearby": "Share of members with heavy rain within 1–2° (~100–200 km) — a near miss in position.",
  "Members place rain differently": "Members agree on rain but not on where it falls (fractions skill score).",
  "850 hPa moisture": "Water vapour and its transport at ~1.5 km, the fuel for heavy rain.",
  "Low-level convergence": "Air piling together near the surface, forcing it to rise and rain.",
  "Low-level spin (850 hPa vorticity)": "Cyclonic rotation at ~1.5 km, a sign of a developing low or depression.",
  "850 hPa winds": "Wind speed and direction at ~1.5 km (monsoon westerlies, easterly waves).",
  "Sea-level pressure": "Surface pressure pattern: lows, troughs and their depth.",
  "500 hPa pattern": "Mid-level (~5.5 km) height pattern: troughs and ridges steering weather systems.",
  // heatwave
  "Forecast Tmax": "Ensemble-mean daily maximum temperature (bias-corrected).",
  "Forecast heat anomaly": "Forecast Tmax minus the normal for that date and place.",
  "Members disagree on Tmax": "Spread of Tmax across members; high spread means an uncertain forecast.",
  "GEFS heatwave probability": "Share of members meeting the IMD heatwave rule (≥ 40 °C and ≥ 4.5 °C above normal).",
  "Observed heat, last 2 days": "How far above normal the observed Tmax was on the two days before issue.",
  "Normal Tmax for the date": "Climatological maximum temperature for this date and place.",
  "2 m temperature": "Near-surface air temperature forecast.",
  "850 hPa temperature": "Temperature at ~1.5 km; warm air aloft favours extreme surface heat.",
  "Upper-level ridge (500/300 hPa)": "High-pressure dome aloft that traps heat and suppresses cloud.",
  "Near-surface humidity": "Moisture near the ground; dry air heats faster, moist air can bring storms.",
  // low-pressure systems
  "GEFS storm within 600 km": "Share of members tracking a low or depression within 300–600 km.",
  "Spin in GEFS members (850 hPa)": "Members' cyclonic rotation at ~1.5 km, mean and spread.",
  "Pressure dip in GEFS members": "How deep a pressure low the members show compared with surroundings.",
  "Systems GEFS sees in the domain": "Number of lows the members track anywhere over India and nearby seas.",
  "MJO at issue time": "Madden–Julian Oscillation phase and strength; an active MJO favours monsoon lows.",
  "Seasonal storm climatology": "How often a low passes here at this time of year.",
  // western disturbances
  "Forecast precipitation": "Ensemble-mean 24 h precipitation (rain and snow).",
  "Members disagree on precipitation": "Spread of precipitation across members.",
  "GEFS heavy-precipitation probability": "Share of members forecasting heavy (≥ 35.6 mm) or moderate precipitation.",
  "Heavy precipitation forecast nearby": "Chance of heavy precipitation in the neighbouring cells.",
  "WD in GEFS members within 600 km": "Share of members tracking a western disturbance within 600 km.",
  "500 hPa trough in GEFS": "Mid-level (~5.5 km) trough and spin carried by the westerlies.",
  "WD approaching from the west (ERA5 at issue)": "A trough over Iran/Afghanistan (40–65° E) on the issue day, before it reaches India.",
  "Observed precipitation yesterday": "IMD gridded precipitation on the day before issue.",
  // monsoon
  "GEFS core-zone rain": "Forecast rain over the monsoon core zone and its heavy-rain probabilities.",
  "Ensemble regime signal": "Share of members forecasting an active or a break spell.",
  "Low-level moisture": "Humidity at 850/700 hPa and near the surface, plus moisture transport.",
  "Monsoon flow (850/700 hPa)": "Strength of the monsoon westerlies and low-level jet.",
  "Convection and cloud (OLR, CAPE)": "Outgoing longwave radiation (cloudiness) and instability.",
  "Monsoon trough (pressure, spin, convergence)": "Position and depth of the monsoon trough.",
  "Upper-level pattern": "Mid- and upper-level heights and winds (e.g. tropical easterly jet).",
  "Temperature profile": "Temperatures from the surface to 300 hPa.",
};

export function driverHint(name: string): string {
  return GLOSSARY[name] ?? GLOSSARY[name.replaceAll("NEPS", "GEFS")] ?? name;
}
