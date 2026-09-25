"use client";

import dynamic from "next/dynamic";
import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { DirectionBadge } from "@/components/desk/panels";
import { formatInit, formatValid, pct, TARGET_LABELS, tierSentence } from "@/lib/format";
import { calibrate, level } from "@/lib/risk";
import type { Direction, Hazard, MonsoonData } from "@/lib/types";

const RiskMap = dynamic(() => import("@/components/desk/risk-map").then((m) => m.RiskMap), {
  ssr: false, loading: () => <Skeleton className="h-[min(68vh,560px)] min-h-[340px] w-full" />,
});

const MCZ = { south: 18, north: 28, west: 73, east: 86 };
const dirOf = (t: string): Direction => (t.startsWith("miss") ? "miss" : "false_alarm");
const LEVEL = ["Low", "Elevated", "High"] as const;

const config = {
  miss_break: { label: TARGET_LABELS.miss_break, color: "oklch(0.78 0.14 60)" },
  fa_break: { label: TARGET_LABELS.fa_break, color: "oklch(0.72 0.12 250)" },
  miss_active: { label: TARGET_LABELS.miss_active, color: "var(--miss)" },
  fa_active: { label: TARGET_LABELS.fa_active, color: "var(--fa)" },
} satisfies ChartConfig;

export function MonsoonView({ hazard, data, lead, showObserved }: { hazard: Hazard; data?: MonsoonData; lead: number; showObserved: boolean }) {
  if (!data) return <Skeleton className="h-96 w-full" />;
  const L = lead - 1;
  const levels = data.targets.map((t, j) => (hazard.tiers[t] ? level(data.p[L][j], hazard.tiers[t]) : null));
  const worst = data.targets
    .map((t, j) => ({ t, lv: levels[j] ?? 0, p: data.p[L][j] }))
    .sort((a, b) => b.lv - a.lv || b.p - a.p)[0];
  const rows = Array.from({ length: 9 }, (_, i) =>
    Object.fromEntries([["lead", i + 1], ...data.targets.map((t, j) => [t, +(data.p[i][j] * 100).toFixed(1)])]));

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {data.targets.map((t, j) => {
          const lv = levels[j], obs = data.obs[L][j];
          return (
            <Card key={t} size="sm" className="gap-2">
              <CardHeader className="flex flex-row items-center justify-between gap-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">{TARGET_LABELS[t]}</CardTitle>
                <DirectionBadge direction={dirOf(t)} />
              </CardHeader>
              <CardContent className="flex flex-col gap-1">
                <p className="text-2xl font-semibold tabular-nums">
                  {hazard.tiers[t] ? pct(calibrate(data.p[L][j], hazard.tiers[t].calibration)) : "—"}
                  <span className="ml-1.5 text-sm font-normal text-muted-foreground">chance</span>
                </p>
                <p className="text-xs text-muted-foreground">
                  {lv === null ? "Not scored — too few validation cases" : <>Alert: <b className="font-medium text-foreground">{LEVEL[lv]}</b></>}
                  {showObserved && obs >= 0 && <> · happened: <b className="font-medium text-foreground">{obs === 1 ? "yes" : "no"}</b></>}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <Card className="gap-3">
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <div className="flex flex-col gap-1">
              <CardTitle>Monsoon core zone · Day {lead}</CardTitle>
              <CardDescription>Valid {formatValid(data.init, lead)} · strongest alert shades the box</CardDescription>
            </div>
            <Badge variant="outline">Regional</Badge>
          </CardHeader>
          <CardContent>
            <RiskMap cells={null} mode="both" showObserved={false} showGefs={false} box={MCZ}
              boxTone={worst && worst.lv > 0 ? dirOf(worst.t) : "neutral"}
              label={`Monsoon core zone, day ${lead}: strongest alert ${worst ? `${LEVEL[worst.lv]} for ${TARGET_LABELS[worst.t]}` : "none"}.`} />
          </CardContent>
        </Card>
        <div className="flex flex-col gap-4">
          <Card size="sm">
            <CardHeader><CardTitle>GEFS regime call</CardTitle></CardHeader>
            <CardContent>
              <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-y-2 text-sm">
                <dt className="text-muted-foreground">Members forecasting an active spell</dt>
                <dd className="text-right tabular-nums">{pct(data.p_active[L])}</dd>
                <dt className="text-muted-foreground">Members forecasting a break spell</dt>
                <dd className="text-right tabular-nums">{pct(data.p_break[L])}</dd>
              </dl>
            </CardContent>
          </Card>
          <Card size="sm">
            <CardHeader><CardTitle>What a high alert means</CardTitle></CardHeader>
            <CardContent>
              <ul className="flex flex-col divide-y text-sm">
                {data.targets.filter((t) => hazard.tiers[t]).map((t) => (
                  <li key={t} className="flex flex-col gap-0.5 py-2.5 first:pt-0">
                    <span className="font-medium">{TARGET_LABELS[t]}</span>
                    <span className="text-pretty text-muted-foreground">{tierSentence(hazard.tiers[t].high, dirOf(t), "spell")}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Regime-bust risk across the 9-day forecast</CardTitle>
          <CardDescription>Issued {formatInit(data.init)}{showObserved ? " · open rings mark days the bust really happened" : ""}</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={config} className="aspect-auto h-64 w-full">
            <LineChart data={rows} margin={{ left: 4, right: 8, top: 8 }} accessibilityLayer>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="lead" tickLine={false} axisLine={false} tickFormatter={(d) => `D${d}`} />
              <YAxis tickLine={false} axisLine={false} width={40} unit="%" />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <ReferenceLine x={lead} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
              {data.targets.map((t, j) => (
                <Line key={t} dataKey={t} type="monotone" stroke={`var(--color-${t})`} strokeWidth={2}
                  strokeDasharray={dirOf(t) === "false_alarm" ? "5 4" : undefined}
                  dot={(p: { cx?: number; cy?: number; index?: number }) => {
                    const hit = showObserved && data.obs[p.index ?? 0]?.[j] === 1;
                    return <circle key={`${t}-${p.index}`} cx={p.cx} cy={p.cy} r={hit ? 6 : 2.5}
                      fill={hit ? "var(--background)" : `var(--color-${t})`} stroke={`var(--color-${t})`} strokeWidth={hit ? 2 : 0} />;
                  }} />
              ))}
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}
