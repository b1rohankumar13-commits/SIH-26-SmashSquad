"use client";

import useSWR from "swr";

import type { CaseStudy, Explanation, GridData, Hazard, LeadRow, MonsoonData, Track } from "@/lib/types";

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

export const useExplain = (hazard?: string, init?: string, lead?: number) =>
  useSWR<Explanation>(hazard && init && lead ? `/api/explain/${hazard}?init=${init}&lead=${lead}` : null, fetcher, opts);
