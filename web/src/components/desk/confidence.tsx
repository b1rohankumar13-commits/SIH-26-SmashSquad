"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cellsLabel, confidence, expectedCells, GRADE_COLOUR, GRADE_LABEL, pctLabel, rgba, type Grade } from "@/lib/risk";
import type { Direction, Explanation, Mode, RegionData } from "@/lib/types";
import { cn } from "@/lib/utils";
import { driverHint } from "@/lib/glossary";

const DAYS = Array.from({ length: 9 }, (_, i) => i + 1);
const GRADES: Grade[] = ["low", "medium", "high"];

export function GradeSwatch({ grade }: { grade: Grade }) {
  return <span aria-hidden className="mt-0.5 size-3.5 shrink-0 rounded-sm ring-1 ring-foreground/20"
    style={{ background: rgba(GRADE_COLOUR[grade]) }} />;
}

/** Legend rows for the confidence view. */
export function ConfidenceLegend() {
  return (
    <ul className="flex flex-col gap-2 text-sm">
      {GRADES.map((g) => (
        <li key={g} className="flex gap-3">
          <GradeSwatch grade={g} />
          <span><b className="font-medium">{GRADE_LABEL[g]} confidence</b>{" "}
            <span className="text-muted-foreground">
              {g === "low" ? "— among the riskiest 10% of forecasts, and ≥ 25% chance of a bust"
                : g === "medium" ? "— riskiest 30%, and ≥ 10% chance of a bust" : "— GEFS is likely to hold"}
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Region-wise forecast confidence for Days 1-9: every covered subdivision, lowest confidence on
 * the selected day first. Confidence = chance the forecast holds (no miss, no false alarm).
 */
export function ConfidenceCard({ data, cuts, lead, showObserved, onLead }: {
  data?: RegionData; cuts?: [number, number]; lead: number; showObserved: boolean; onLead: (lead: number) => void;
}) {
  if (!data || !cuts) return <Skeleton className="h-96 w-full" />;
  const L = lead - 1;
  const rows = data.covered
    .map((r) => ({ r, c: confidence(data.p_miss[L][r], data.p_fa[L][r], cuts).conf, n: expectedCells(data, L, r) }))
    .sort((a, b) => a.c - b.c || b.n - a.n);
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardTitle>Forecast confidence by subdivision · Days 1–9</CardTitle>
        <CardDescription>Chance the GEFS forecast holds (no missed event, no false alarm), lowest on Day {lead} first.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="max-h-[28rem] overflow-auto">
          <table className="w-full min-w-[34rem] border-separate border-spacing-0.5 text-xs tabular-nums">
            <thead className="sticky top-0 bg-card">
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
                    const { conf, grade } = confidence(data.p_miss[d - 1][r], data.p_fa[d - 1][r], cuts);
                    const busted = showObserved && (data.obs_miss[d - 1][r] === 1 || data.obs_fa[d - 1][r] === 1);
                    return (
                      <td key={d} title={`Day ${d}: ${GRADE_LABEL[grade]} confidence (${pctLabel(conf)}) · ${cellsLabel(expectedCells(data, d - 1, r))} expected to bust${busted ? " · busted" : ""}`}
                        className={cn("relative h-7 rounded text-center", d === lead && "ring-1 ring-foreground/40",
                          grade === "low" && "text-white")}
                        style={{ background: grade === "high" ? undefined : rgba(GRADE_COLOUR[grade]) }}>
                        {Math.round(conf * 100)}
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
          Percent confidence{showObserved && "; a dot marks a subdivision where the forecast busted"}.
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Why the least-confident subdivisions are flagged on this day: drivers of the explained areas
 * overlapping each one, weighted by the area's expected busts and its share of cells inside.
 */
export function RegionReasonsCard({ data, explain, cuts, lead, mode }: {
  data?: RegionData; explain?: Explanation; cuts?: [number, number]; lead: number; mode: Mode;
}) {
  if (!data || !explain) return null;
  const L = lead - 1;
  const dirs = (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[];
  const risk = (r: number) => (cuts ? 1 - confidence(data.p_miss[L][r], data.p_fa[L][r], cuts).conf
    : Math.max(mode !== "false_alarm" ? data.p_miss[L][r] : 0, mode !== "miss" ? data.p_fa[L][r] : 0));
  const reasons = data.covered.map((r) => {
    const w: Record<string, { share: number; value: string }> = {};
    let total = 0;
    for (const d of dirs) {
      for (const a of explain[d]) {
        const inside = a.regions?.[String(r)] ?? 0;
        if (!inside) continue;
        const weight = a.expected * inside / a.cells;
        total += weight;
        for (const dr of a.drivers) {
          const cur = w[dr.name];
          w[dr.name] = { share: (cur?.share ?? 0) + weight * dr.share, value: cur?.value ?? dr.value };
        }
      }
    }
    const drivers = Object.entries(w).map(([name, v]) => ({ name, share: v.share / (total || 1), value: v.value }))
      .sort((a, b) => b.share - a.share).slice(0, 3);
    return { r, p: risk(r), n: expectedCells(data, L, r, mode), drivers };
  }).filter((x) => x.drivers.length).sort((a, b) => b.p - a.p || b.n - a.n).slice(0, 4);
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Why these subdivisions?</CardTitle>
        <CardDescription>Main reasons behind the least reliable subdivisions on Day {lead}.</CardDescription>
      </CardHeader>
      <CardContent>
        {reasons.length === 0 ? (
          <p className="text-sm text-muted-foreground">No subdivision has a flagged area on this day.</p>
        ) : (
          <ul className="flex flex-col divide-y">
            {reasons.map(({ r, p, n, drivers }) => (
              <li key={r} className="flex flex-col gap-1.5 py-2.5 first:pt-0 last:pb-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-medium">{data.names[r]}</span>
                  <span className="text-xs text-muted-foreground tabular-nums">{pctLabel(p)} bust chance · {cellsLabel(n)}</span>
                </div>
                {drivers.map((d) => (
                  <div key={d.name} className="grid grid-cols-[minmax(0,1fr)_auto_3rem] items-center gap-2 text-xs">
                    <span className="truncate cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title={driverHint(d.name)}>{d.name}</span>
                    <span className="text-muted-foreground tabular-nums">{d.value}</span>
                    <span className="flex items-center gap-1.5 tabular-nums">
                      <span aria-hidden className="h-1.5 rounded-full bg-foreground/70" style={{ width: `${Math.max(4, d.share * 24)}px` }} />
                      {Math.round(d.share * 100)}%
                    </span>
                  </div>
                ))}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
