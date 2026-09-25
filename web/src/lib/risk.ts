import type { Box, Calibration, CurvePoint, Direction, GridData, Mode, Tiers } from "@/lib/types";

/**
 * Raw model scores are inflated by class-weighted training (a 0.97 rain-miss score busts
 * ~10–19% of the time), so everything user-facing uses the calibrated bust probability:
 * the observed validation bust rate at that score (isotonic fit, see export script).
 */
export function calibrate(p: number, cal: Calibration): number {
  const { x, y } = cal;
  if (p <= x[0]) return y[0];
  if (p >= x[x.length - 1]) return y[y.length - 1];
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= p) lo = mid; else hi = mid;
  }
  const t = (p - x[lo]) / (x[hi] - x[lo] || 1);
  return y[lo] + t * (y[hi] - y[lo]);
}

/** Selectable cut-offs, ascending; drops top-end cut-offs that catch <1% of busts (too few cells to trust). */
export function cutoffs(cal: Calibration): CurvePoint[] {
  return cal.curve.filter((c) => c.p > 0 && c.catch >= 0.01).sort((a, b) => a.p - b.p);
}

export type Thresholds = Record<Direction, number>;

export interface Cell {
  lat: number;
  lon: number;
  /** Calibrated bust probabilities. */
  pMiss: number;
  pFa: number;
  obsMiss: boolean;
  obsFa: boolean;
  ens: number;
  missFlag: boolean;
  faFlag: boolean;
  /** Shown on the map for the current mode (null = too close to the normal rate to colour). */
  direction: Direction | null;
  /** 0-1 colour intensity (log scale from the normal bust rate to the highest validated chance). */
  strength: number;
}

/**
 * Colour scale for a direction: intensity grows with log(p / base rate), so a cell at the
 * normal bust rate is invisible and the highest validated chance is darkest.
 */
export interface Scale { base: number; cap: number }

export const scaleFor = (t: Tiers): Scale => ({
  base: t.base_rate,
  cap: Math.max(cutoffs(t.calibration).at(-1)?.p ?? t.calibration.max, t.base_rate * 2),
});

export const intensity = (p: number, s: Scale) =>
  p <= s.base ? 0 : Math.min(1, Math.log(p / s.base) / Math.log(s.cap / s.base));

/** Value at a given intensity (for legend ticks). */
export const atIntensity = (t: number, s: Scale) => s.base * Math.exp(t * Math.log(s.cap / s.base));

/**
 * Only the upper half of the log scale is drawn (≈10× the normal rate and up, e.g. ≥4.5% for
 * rain misses); lower chances add noise, not signal. The colour ramp spans the shown half.
 */
export const SHOW_FROM = 0.5;

export function classify(g: GridData, tiers: Record<string, Tiers>, mode: Mode, thr: Thresholds): Cell[] {
  const cm = tiers.miss.calibration, cf = tiers.false_alarm.calibration;
  const sm = scaleFor(tiers.miss), sf = scaleFor(tiers.false_alarm);
  return g.lat.map((lat, i) => {
    const pMiss = calibrate(g.p_miss[i], cm), pFa = calibrate(g.p_fa[i], cf);
    const iM = intensity(pMiss, sm), iF = intensity(pFa, sf);
    let direction: Direction | null = null, strength = 0;
    if (mode === "miss") [direction, strength] = ["miss", iM];
    else if (mode === "false_alarm") [direction, strength] = ["false_alarm", iF];
    else [direction, strength] = iM >= iF ? ["miss", iM] : ["false_alarm", iF];
    if (strength < SHOW_FROM) direction = null;
    strength = Math.max(0, (strength - SHOW_FROM) / (1 - SHOW_FROM));
    return {
      lat, lon: g.lon[i], pMiss, pFa, ens: g.ens_prob[i],
      obsMiss: g.obs_miss[i] === 1, obsFa: g.obs_fa[i] === 1,
      // Flag = coloured on the map (at or above the display cut-off).
      missFlag: pMiss >= thr.miss, faFlag: pFa >= thr.false_alarm,
      direction, strength,
    };
  });
}

export const inBox = (c: { lat: number; lon: number }, b: Box) =>
  c.lat >= b.south && c.lat <= b.north && c.lon >= b.west && c.lon <= b.east;

/** Colour ramps from barely-above-normal (pale, translucent) to most likely (dark). Misses orange, false alarms blue. */
export const RAMP: Record<Direction, { from: number[]; to: number[]; ringDark: number[]; ringLight: number[] }> = {
  miss: { from: [254, 215, 170, 60], to: [154, 52, 18, 240], ringDark: [255, 237, 213, 255], ringLight: [67, 20, 7, 255] },
  false_alarm: { from: [191, 219, 254, 55], to: [30, 58, 138, 235], ringDark: [219, 234, 254, 255], ringLight: [15, 23, 42, 255] },
};

export function rampColour(d: Direction, t: number): [number, number, number, number] {
  const { from, to } = RAMP[d];
  return from.map((f, k) => Math.round(f + (to[k] - f) * t)) as [number, number, number, number];
}

/** "0.04%", "4.5%", "45%" - enough precision for tiny calibrated probabilities. */
export const pctLabel = (p: number) =>
  p < 0.001 ? `${(p * 100).toFixed(2)}%` : p < 0.1 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`;

export const rgba = (c: readonly number[]) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${(c[3] / 255).toFixed(2)})`;

/** Kept for the regional (monsoon) view, which still uses the validation tiers. */
export type Level = 0 | 1 | 2;
export function level(p: number, t: Tiers): Level {
  if (p >= t.high.threshold) return 2;
  if (t.elevated && p >= t.elevated.threshold) return 1;
  return 0;
}
