"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { pctLabel, rampColour, regionTone, rgba } from "@/lib/risk";
import type { Direction, Hazard, Mode, RegionData } from "@/lib/types";
import { cn } from "@/lib/utils";

const TOP = 8;
const DAYS = Array.from({ length: 9 }, (_, i) => i + 1);
const LABEL: Record<Direction, string> = { miss: "Misses", false_alarm: "False alarms" };

/** Bust chance shown for a subdivision on one day under the current mode. */
function chance(d: RegionData, L: number, r: number, mode: Mode) {
  const m = d.p_miss[L][r], f = d.p_fa[L][r];
  return mode === "miss" ? m : mode === "false_alarm" ? f : Math.max(m, f);
}

/**
 * Subdivisions ranked by bust chance on the selected day, with Days 1-9 alongside:
 * the region-wise confidence view (low chance = GEFS can be trusted there).
 */
export function RegionCard({ hazard, data, mode, lead, showObserved, onLead }: {
  hazard: Hazard; data?: RegionData; mode: Mode; lead: number; showObserved: boolean; onLead: (lead: number) => void;
}) {
  if (!data) return <Skeleton className="h-72 w-full" />;
  const L = lead - 1;
  const rows = data.covered
    .map((r) => ({ r, p: chance(data, L, r, mode) }))
    .sort((a, b) => b.p - a.p)
    .slice(0, TOP);
  const skill = hazard.regions?.skill;
  const dirs = (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[];
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Most error-prone subdivisions · Day {lead}</CardTitle>
        <CardDescription>Chance GEFS busts somewhere in the subdivision, Days 1–9.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] border-separate border-spacing-0.5 text-xs tabular-nums">
            <thead>
              <tr className="text-muted-foreground">
                <th scope="col" className="pr-2 text-left font-normal">Subdivision</th>
                {DAYS.map((d) => (
                  <th key={d} scope="col" className="w-11 font-normal">
                    <button type="button" onClick={() => onLead(d)} aria-pressed={d === lead}
                      className={cn("w-full rounded px-1 py-0.5 hover:bg-muted", d === lead && "bg-muted font-medium text-foreground")}>
                      D{d}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(({ r }) => (
                <tr key={r}>
                  <th scope="row" className="max-w-[14rem] truncate pr-2 text-left font-normal" title={data.names[r]}>{data.names[r]}</th>
                  {DAYS.map((d) => {
                    const m = data.p_miss[d - 1][r], f = data.p_fa[d - 1][r];
                    const tone = regionTone(m, f, data.flag_miss[d - 1][r] === 1, data.flag_fa[d - 1][r] === 1, mode);
                    const busted = showObserved && dirs.some((k) => (k === "miss" ? data.obs_miss : data.obs_fa)[d - 1][r] === 1);
                    return (
                      <td key={d} title={`Day ${d}: miss ${pctLabel(m)} · false alarm ${pctLabel(f)}${busted ? " · busted" : ""}`}
                        className={cn("relative h-7 rounded text-center", d === lead && "ring-1 ring-foreground/40",
                          tone && tone.strength > 0.55 && "text-white")}
                        style={{ background: tone ? rgba(rampColour(tone.direction, tone.strength)) : undefined }}>
                        {Math.round(chance(data, d - 1, r, mode) * 100)}
                        {busted && <span aria-label="busted" className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-foreground" />}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-pretty text-muted-foreground">
          Percent chance; shaded where the grid map flags cells{showObserved && ", dot = busted that day"}.
          {skill && " "}
          {skill && dirs.map((k) => `${LABEL[k]}: AUROC ${skill[k].model.auroc.toFixed(2)} vs ${skill[k].baseline.auroc.toFixed(2)} for GEFS alone`).join(" · ")}
          {skill && "."}
        </p>
      </CardContent>
    </Card>
  );
}
