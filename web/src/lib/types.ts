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
}

export interface Tiers {
  base_rate: number;
  pr_auc: number;
  high: Tier;
  elevated: Tier | null;
  calibration: Calibration;
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
  targets?: string[];
  region?: string;
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
}
