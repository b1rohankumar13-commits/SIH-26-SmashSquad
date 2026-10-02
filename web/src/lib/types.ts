export type Direction = "miss" | "false_alarm";
export type Mode = Direction | "both";

export interface Tier {
  threshold: number;
  catch_rate: number;
  precision: number;
  lift: number | null;
  flagged_share: number;
}

export interface CurvePoint {
  /** Calibrated-probability cut-off: flag cells at or above it. */
  p: number;
  /** Share of validation busts caught at this cut-off. */
  catch: number;
  /** Share of flagged validation cells that really busted. */
  precision: number;
  /** Share of all validation cells flagged. */
  share: number;
}

export interface Calibration {
  /** Raw model score knots → observed bust rate (isotonic, validation set). */
  x: number[];
  y: number[];
  max: number;
  curve: CurvePoint[];
  /** Per lead-band knots (days 1-3 / 4-6 / 7-9); the pooled x/y is the fallback. */
  bands?: { leads: number[]; x: number[]; y: number[] }[];
}

export interface Tiers {
  base_rate: number;
  pr_auc: number;
  high: Tier;
  elevated: Tier | null;
  calibration: Calibration;
}

export interface Scores {
  auroc: number;
  pr_auc: number;
  /** Brier skill vs climatology, isotonic-calibrated on the other half of the validation period. */
  bss: number;
}

/** Model vs the ensemble-only logistic baseline, on identical validation cells. */
export interface Skill {
  model: Scores;
  baseline: Scores;
  n: number;
  busts: number;
}

export interface Hazard {
  id: string;
  label: string;
  kind: "grid" | "regional";
  event: string;
  model: string;
  truth: string;
  val_period: string;
  n_inits: number;
  first_init: string;
  last_init: string;
  notable_inits?: string[];
  tiers: Record<string, Tiers>;
  skill?: Record<string, Skill>;
  targets?: string[];
  region?: string;
  /** Days added to init + lead to get the valid date: -1 for daily totals (default), 0 for 00 UTC snapshots. */
  valid_shift?: number;
  /** Subdivision-level risk (scripts/export_regions.py): which subdivisions have observed cells, and skill vs GEFS alone. */
  regions?: { covered: number[]; confidence_cuts?: [number, number]; skill: Record<Direction, { model: Pick<Scores, "auroc" | "pr_auc">; baseline: Pick<Scores, "auroc" | "pr_auc">; base_rate: number; n: number }> };
}

/** Chance GEFS busts somewhere in each IMD subdivision; arrays are [lead][region], obs -1 = no observed cell. */
export interface RegionData {
  init: string;
  names: string[];
  covered: number[];
  p_miss: number[][];
  p_fa: number[][];
  obs_miss: number[][];
  obs_fa: number[][];
  /** 1 where the grid map colours a cell in the subdivision; only those are shaded. */
  flag_miss: number[][];
  flag_fa: number[][];
  /** Expected busting cells (sum of calibrated cell risks); ranks subdivisions whose chances tie. */
  exp_miss?: number[][];
  exp_fa?: number[][];
}

export type View = "cells" | "regions" | "confidence";

/** A flagged area on one map and the drivers that push its risk up (gradient x input, see export_explanations.py). */
export interface ExplainedArea {
  lat: number;
  lon: number;
  cells: number;
  /** Sum of the calibrated bust probability over the area's cells. */
  expected: number;
  /** [south, north, west, east] */
  box: [number, number, number, number];
  drivers: { name: string; share: number; value: string }[];
  /** Subdivision id -> this area's cells inside it. */
  regions?: Record<string, number>;
}

export type Explanation = Record<Direction, ExplainedArea[]>;

/** An observed (ERA5) low-pressure-system track near the valid date; points are [lon, lat]. */
export interface Track {
  track: number;
  path: [number, number][];
  dates: string[];
  now: [number, number];
  first: string;
  last: string;
}

export interface GridData {
  init: string;
  lead: number;
  lat: number[];
  lon: number[];
  p_miss: number[];
  p_fa: number[];
  obs_miss: number[];
  obs_fa: number[];
  ens_prob: number[];
}

export interface LeadRow {
  lead: number;
  miss_flagged: number;
  miss_observed: number;
  false_alarm_flagged: number;
  false_alarm_observed: number;
}

export interface MonsoonData {
  init: string;
  targets: string[];
  p: number[][];
  obs: number[][];
  p_active: number[];
  p_break: number[];
}

export interface Box {
  south: number;
  north: number;
  west: number;
  east: number;
}

export interface CaseStudy {
  id: string;
  title: string;
  hazard: string;
  init: string;
  lead: number;
  mode: Direction;
  region: string;
  box: Box;
  story: string;
  valid_shift?: number;
}

export type LiveSource = "gefs" | "neps";

/** One hazard in a live run (scripts/live_forecast.py). */
export interface LiveHazardRun {
  status: "ok" | "out_of_season" | "missing_input";
  /** Real-time stand-ins used for observations that are not available yet. */
  substitutions?: string[];
  /** Truth for the valid dates was on disk, so observed busts are included. */
  validated?: boolean;
  valid_shift?: number;
  validation?: Record<Direction, { busts: number; flagged: number; caught: number }>;
  missing?: string[];
}

export interface LiveRun {
  source: LiveSource;
  source_label: string;
  init: string;
  created: string;
  realtime: boolean;
  hazards: Record<string, LiveHazardRun>;
}

export interface LiveJob {
  job: string;
  status: "running" | "done" | "failed";
  source: LiveSource;
  init: string | null;
  log: string[];
}
