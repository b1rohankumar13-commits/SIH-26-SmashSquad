"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useTransition } from "react";

import type { Mode } from "@/lib/types";

export interface DeskState {
  hazard: string;
  mode: Mode;
  init?: string;
  lead: number;
  observed: boolean;
  gefs: boolean;
  caseId?: string;
}

const MODES: Mode[] = ["both", "miss", "false_alarm"];

/** All desk filters live in the URL (shareable, refresh-safe, back/forward-aware). */
export function useDeskState() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [pending, startTransition] = useTransition();

  const lead = Number(params.get("lead") ?? 1);
  const modeParam = params.get("mode") as Mode | null;
  const state: DeskState = {
    hazard: params.get("hazard") ?? "rain",
    mode: modeParam && MODES.includes(modeParam) ? modeParam : "both",
    init: params.get("init") ?? undefined,
    lead: Number.isInteger(lead) && lead >= 1 && lead <= 9 ? lead : 1,
    observed: params.get("observed") !== "0",
    gefs: params.get("gefs") === "1",
    caseId: params.get("case") ?? undefined,
  };

  const update = useCallback(
    (patch: Partial<DeskState>, { push = false }: { push?: boolean } = {}) => {
      const next = new URLSearchParams(params.toString());
      const set = (k: string, v: string | undefined) => (v === undefined ? next.delete(k) : next.set(k, v));
      if ("hazard" in patch) set("hazard", patch.hazard);
      if ("mode" in patch) set("mode", patch.mode === "both" ? undefined : patch.mode);
      if ("init" in patch) set("init", patch.init);
      if ("lead" in patch) set("lead", patch.lead === 1 ? undefined : String(patch.lead));
      if ("observed" in patch) set("observed", patch.observed ? undefined : "0");
      if ("gefs" in patch) set("gefs", patch.gefs ? "1" : undefined);
      // Hazard/date/lead/direction changes leave a case study unless the patch sets one;
      // overlay toggles keep it.
      const display = Object.keys(patch).every((k) => ["observed", "gefs"].includes(k));
      if (!display || "caseId" in patch) set("case", "caseId" in patch ? patch.caseId : undefined);
      const url = `${pathname}?${next.toString()}`;
      startTransition(() => (push ? router.push(url, { scroll: false }) : router.replace(url, { scroll: false })));
    },
    [params, pathname, router],
  );

  return { state, update, pending };
}
