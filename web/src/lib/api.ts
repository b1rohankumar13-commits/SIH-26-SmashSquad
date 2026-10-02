"use client";

import useSWR from "swr";

import type { CaseStudy, Explanation, GridData, Hazard, LeadRow, LiveJob, LiveRun, LiveSource, MonsoonData, RegionData, Track } from "@/lib/types";

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const opts = { revalidateOnFocus: false, keepPreviousData: true } as const;

export const useHazards = () => useSWR<Hazard[]>("/api/hazards", fetcher, opts);
export const useCases = () => useSWR<CaseStudy[]>("/api/cases", fetcher, opts);

export const useInits = (hazard?: string) =>
  useSWR<{ inits: string[]; notable: string[] }>(
    hazard ? `/api/hazards/${hazard}/inits` : null, fetcher, opts);

export const useGrid = (hazard?: string, init?: string, lead?: number) =>
  useSWR<GridData>(
    hazard && init && lead ? `/api/grid/${hazard}?init=${init}&lead=${lead}` : null, fetcher, opts);

export const useLeads = (hazard?: string, init?: string) =>
  useSWR<LeadRow[]>(hazard && init ? `/api/grid/${hazard}/leads?init=${init}` : null, fetcher, opts);

export const useMonsoon = (init?: string) =>
  useSWR<MonsoonData>(init ? `/api/monsoon?init=${init}` : null, fetcher, opts);

export const useTracks = (hazard?: string, date?: string) =>
  useSWR<Track[]>(hazard && date ? `/api/tracks?hazard=${hazard}&date=${date}` : null, fetcher, opts);

export const useSubdivisions = (enabled: boolean) =>
  useSWR<GeoJSON.FeatureCollection>(enabled ? "/api/subdivisions" : null, fetcher, { ...opts, revalidateIfStale: false });

export const useRegions = (hazard?: string, init?: string) =>
  useSWR<RegionData>(hazard && init ? `/api/regions/${hazard}?init=${init}` : null, fetcher, opts);

/** Monsoon: drivers per flagged regime target at one lead. */
export const useMonsoonExplain = (init?: string, lead?: number) =>
  useSWR<Record<string, { name: string; share: number; value: string }[]>>(
    init && lead ? `/api/explain/monsoon?init=${init}&lead=${lead}` : null, fetcher, opts);

export const useExplain = (hazard?: string, init?: string, lead?: number) =>
  useSWR<Explanation>(hazard && init && lead ? `/api/explain/${hazard}?init=${init}&lead=${lead}` : null, fetcher, opts);

// ---------------------------------------------------------------- live forecast analysis
export const useLiveRuns = () => useSWR<LiveRun[]>("/api/live/runs", fetcher, opts);

const liveBase = (source?: string, init?: string) => (source && init ? `/api/live/${source}/${init}` : null);

export const useLiveGrid = (source?: string, init?: string, hazard?: string, lead?: number) =>
  useSWR<GridData>(liveBase(source, init) && hazard && lead ? `${liveBase(source, init)}/grid/${hazard}?lead=${lead}` : null, fetcher, opts);

export const useLiveRegions = (source?: string, init?: string, hazard?: string) =>
  useSWR<RegionData>(liveBase(source, init) && hazard ? `${liveBase(source, init)}/regions/${hazard}` : null, fetcher, opts);

export const useLiveExplain = (source?: string, init?: string, hazard?: string, lead?: number) =>
  useSWR<Explanation>(liveBase(source, init) && hazard && lead ? `${liveBase(source, init)}/explain/${hazard}?lead=${lead}` : null, fetcher, opts);

/** Polls every 2 s while the job runs. */
export const useLiveJob = (job?: string) =>
  useSWR<LiveJob>(job ? `/api/live/jobs/${job}` : null, fetcher,
    { refreshInterval: (d) => (d && d.status !== "running" ? 0 : 2000) });

export async function startLiveRun(body: { source: LiveSource; init?: string; fetch: boolean; realtime: boolean; demo?: boolean }) {
  const res = await fetch("/api/live/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `${res.status} ${res.statusText}`);
  return (await res.json()) as { job: string };
}

export interface LiveAuto {
  enabled: boolean;
  last_check: string | null;
  last_job: string | null;
  last_init: string | null;
  every_minutes: number;
  next_expected_init: string;
}

export const useLiveAuto = () => useSWR<LiveAuto>("/api/live/auto", fetcher, { refreshInterval: 60_000 });

export async function setLiveAuto(enabled: boolean) {
  const res = await fetch("/api/live/auto", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ enabled }) });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as LiveAuto;
}
