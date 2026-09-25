"use client";

import { Bar, CartesianGrid, ComposedChart, Line, ReferenceLine, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import { formatInit, formatValidShort, oneIn, pct } from "@/lib/format";
import { atIntensity, inBox, pctLabel, rampColour, rgba, SHOW_FROM, type Cell, type Scale, type Thresholds } from "@/lib/risk";
import type { CaseStudy, Direction, Hazard, LeadRow, Mode } from "@/lib/types";
import { cn } from "@/lib/utils";

export function DirectionBadge({ direction }: { direction: Direction }) {
  return direction === "miss"
    ? <Badge className="border-transparent bg-miss-soft text-miss">Under-forecast</Badge>
    : <Badge className="border-transparent bg-fa-soft text-fa">Over-forecast</Badge>;
}

function Metric({ label, value, hint, badge }: { label: string; value: React.ReactNode; hint: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <Card size="sm" className="gap-2">
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2">
        <CardTitle className="text-sm font-medium whitespace-nowrap text-muted-foreground">{label}</CardTitle>
        {badge}
      </CardHeader>
      <CardContent className="flex flex-col gap-1">
        <p className="text-2xl font-semibold tracking-tight tabular-nums">{value}</p>
        <p className="text-xs text-pretty text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

const Small = ({ children }: { children: React.ReactNode }) => (
  <span className="text-sm font-normal text-muted-foreground">{children}</span>
);

export function MetricsSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {Array.from({ length: 4 }, (_, i) => (
        <Card key={i} size="sm" className="gap-2">
          <CardHeader><Skeleton className="h-4 w-24" /></CardHeader>
          <CardContent className="flex flex-col gap-2"><Skeleton className="h-8 w-32" /><Skeleton className="h-3 w-40" /></CardContent>
        </Card>
      ))}
    </div>
  );
}

export function GridMetrics({ cells, thresholds, event, validLabel }: { cells: Cell[]; thresholds: Thresholds; event: string; validLabel: string }) {
  const n = (f: (c: Cell) => boolean) => cells.filter(f).length;
  const obsM = n((c) => c.obsMiss), obsF = n((c) => c.obsFa);
  const caughtM = n((c) => c.obsMiss && c.missFlag), caughtF = n((c) => c.obsFa && c.faFlag);
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Metric label="Miss risk" badge={<DirectionBadge direction="miss" />} value={<>{n((c) => c.missFlag)} <Small>cells</Small></>}
        hint={<>≥ {pctLabel(thresholds.miss)} chance GEFS is <b className="font-medium text-foreground">under-calling</b> {event} on {validLabel}</>} />
      <Metric label="False-alarm risk" badge={<DirectionBadge direction="false_alarm" />} value={<>{n((c) => c.faFlag)} <Small>cells</Small></>}
        hint={<>≥ {pctLabel(thresholds.false_alarm)} chance GEFS is <b className="font-medium text-foreground">over-calling</b> {event}</>} />
      <Metric label="GEFS says" value={<>{n((c) => c.ens >= 0.5)} <Small>cells</Small></>}
        hint={`≥ 50% of members forecast ${event} (of ${cells.length.toLocaleString("en-IN")} cells)`} />
      <Metric label="What happened" badge={<Badge variant="outline">Replay</Badge>}
        value={<>{obsM} <Small>missed</Small> · {obsF} <Small>false</Small></>}
        hint={obsM || obsF ? `Alerts covered ${caughtM}/${obsM} misses and ${caughtF}/${obsF} false alarms` : "The forecast held everywhere at this lead"} />
    </div>
  );
}

function Swatch({ colour }: { colour: readonly number[] }) {
  return <span aria-hidden className="mt-0.5 size-3.5 shrink-0 rounded-[4px]" style={{ background: rgba(colour) }} />;
}

/** Observed-bust ring; theme tokens keep it visible on light and dark cards. */
function RingSwatch({ direction }: { direction: Direction }) {
  return <span aria-hidden className={cn("mt-0.5 size-3.5 shrink-0 rounded-full border-2", direction === "miss" ? "border-miss" : "border-fa")} />;
}

function Ramp({ direction }: { direction: Direction }) {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => rgba(rampColour(direction, t)));
  return <span aria-hidden className="mt-1 h-3 w-full rounded-full ring-1 ring-foreground/10" style={{ background: `linear-gradient(90deg, ${stops.join(", ")})` }} />;
}

export function MapLegend({ scales, mode, event, showObserved, showGefs }: {
  scales: Record<Direction, Scale>; mode: Mode; event: string; showObserved: boolean; showGefs: boolean;
}) {
  const dirs = (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[];
  return (
    <Card size="sm">
      <CardHeader><CardTitle>How to read the map</CardTitle></CardHeader>
      <CardContent>
        <ul className="flex flex-col divide-y text-sm">
          {dirs.map((d) => (
            <li key={d} className="flex flex-col gap-1 py-2.5 first:pt-0">
              <span className="font-medium">{d === "miss" ? "Likely miss" : "Likely false alarm"}</span>
              <Ramp direction={d} />
              <span className="flex justify-between text-xs text-muted-foreground tabular-nums">
                <span>{pctLabel(atIntensity(SHOW_FROM, scales[d]))}</span>
                <span>{pctLabel(atIntensity((1 + SHOW_FROM) / 2, scales[d]))}</span>
                <span>{pctLabel(scales[d].cap)}+</span>
              </span>
              <span className="text-pretty text-muted-foreground">
                {d === "miss" ? `Chance that ${event} arrives with no GEFS warning. Darker = likelier.`
                  : `Chance that the ${event} GEFS forecasts won’t happen. Darker = likelier.`}{" "}
                Uncoloured cells are below {pctLabel(atIntensity(SHOW_FROM, scales[d]))} — about {Math.round(atIntensity(SHOW_FROM, scales[d]) / scales[d].base)}× the normal {pctLabel(scales[d].base)} rate.
              </span>
            </li>
          ))}
          {showObserved && dirs.map((d) => (
            <li key={`obs-${d}`} className="flex gap-3 py-2.5">
              <RingSwatch direction={d} />
              <div className="flex flex-col gap-0.5">
                <span className="font-medium">Observed {d === "miss" ? "miss" : "false alarm"}</span>
                <span className="text-muted-foreground">Where the forecast really busted this way.</span>
              </div>
            </li>
          ))}
          {showGefs && (
            <li className="flex gap-3 py-2.5">
              <Swatch colour={[20, 184, 166, 170]} />
              <div className="flex flex-col gap-0.5">
                <span className="font-medium">GEFS event probability</span>
                <span className="text-muted-foreground">Darker = more of the 5 members forecast {event}.</span>
              </div>
            </li>
          )}
        </ul>
      </CardContent>
    </Card>
  );
}

/** Export strings use ASCII ("Tmax >= 40 C", "0.25 deg"); render proper symbols. */
const prettyText = (t: string) =>
  t.replaceAll(">=", "≥").replace(/(\d) C\b/g, "$1\u00a0°C").replace(/(\d) deg\b/g, "$1°").replace(/(\d) mm\b/g, "$1\u00a0mm");
const prettyPeriod = (t: string) =>
  t.replace(/(\d{8})\.\.(\d{8})/, (_, a: string, b: string) => `${formatInit(a)} – ${formatInit(b)}`);

export function ModelFacts({ hazard, falseAlarmNote }: { hazard: Hazard; falseAlarmNote?: string }) {
  const rows: [string, React.ReactNode][] = [
    ["Event", prettyText(hazard.event)], ["Model", hazard.model], ["Truth", prettyText(hazard.truth)],
    ["Validation", prettyPeriod(hazard.val_period)],
  ];
  for (const [key, t] of Object.entries(hazard.tiers)) {
    rows.push([`${key.replaceAll("_", " ")} skill (PR-AUC)`,
      <span key={key} className="tabular-nums">{t.pr_auc.toFixed(3)} · {Math.round(t.pr_auc / t.base_rate)}× chance</span>]);
  }
  return (
    <Card size="sm">
      <CardHeader><CardTitle>About this model</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-3">
        <dl className="flex flex-col divide-y text-sm">
          {rows.map(([k, v]) => (
            <div key={k} className="flex flex-col gap-0.5 py-2 first:pt-0">
              <dt className="text-xs text-muted-foreground first-letter:uppercase">{k}</dt>
              <dd className="text-pretty">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
          Historical validation period — the model never saw these dates during training.
        </p>
        {falseAlarmNote && <p className="rounded-md bg-fa-soft px-3 py-2 text-xs text-pretty text-fa">{falseAlarmNote}</p>}
      </CardContent>
    </Card>
  );
}

const leadConfig = {
  miss_flagged: { label: "Miss · high alerts", color: "var(--miss)" },
  miss_observed: { label: "Miss · observed", color: "var(--miss)" },
  false_alarm_flagged: { label: "False alarm · high alerts", color: "var(--fa)" },
  false_alarm_observed: { label: "False alarm · observed", color: "var(--fa)" },
} satisfies ChartConfig;

export function LeadChart({ rows, mode, showObserved, lead, init }: { rows?: LeadRow[]; mode: Mode; showObserved: boolean; lead: number; init: string }) {
  const dirs = (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Across the 9-day forecast</CardTitle>
        <CardDescription>Cells at high alert per lead day (bars) against busts that really happened (dashed), issued {formatInit(init)}</CardDescription>
      </CardHeader>
      <CardContent>
        {!rows ? <Skeleton className="h-64 w-full" /> : (
          <ChartContainer config={leadConfig} className="aspect-auto h-64 w-full">
            <ComposedChart data={rows} margin={{ left: 4, right: 8, top: 8 }} accessibilityLayer>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="lead" tickLine={false} axisLine={false} tickMargin={8}
                tickFormatter={(d) => `D${d} · ${formatValidShort(init, d)}`} fontSize={11} />
              <YAxis tickLine={false} axisLine={false} width={36} allowDecimals={false} />
              <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => `Day ${p?.[0]?.payload?.lead}`} />} />
              <ChartLegend content={<ChartLegendContent />} />
              <ReferenceLine x={lead} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
              {dirs.map((d) => <Bar key={`${d}-bar`} dataKey={`${d}_flagged`} fill={`var(--color-${d}_flagged)`} fillOpacity={0.85} radius={[4, 4, 0, 0]} />)}
              {showObserved && dirs.map((d) => (
                <Line key={`${d}-line`} dataKey={`${d}_observed`} type="monotone" stroke={`var(--color-${d}_observed)`}
                  strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
              ))}
            </ComposedChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}

export function CaseCard({ c, cells, event, lead }: { c: CaseStudy; cells: Cell[]; event: string; lead: number }) {
  const box = cells.filter((x) => inBox(x, c.box));
  const gefs = box.filter((x) => x.ens >= 0.5).length;
  const before = `issued ${formatInit(c.init)}, ${lead - 1} days before`;
  let stats: [string, string, string][];
  if (c.mode === "miss") {
    const hit = box.filter((x) => x.obsMiss);
    const caught = hit.filter((x) => x.missFlag).length;
    const top = [...hit].sort((a, b) => b.pMiss - a.pMiss)[0];
    stats = [
      [`GEFS forecast for ${c.region}`, `${gefs} cells`, gefs === 0
        ? `with ≥ 50% of members calling ${event} — effectively no warning`
        : `with ≥ 50% of members calling ${event} — but none where it struck: not one member forecast it in any of the ${hit.length} hit cells`],
      ["What happened", `${hit.length} cells`, `of ${c.region} got ${event} GEFS missed entirely`],
      ["BustSentinel alert", `${caught} of ${hit.length} caught`,
        `${box.filter((x) => x.missFlag).length} ${c.region} cells flagged in total${top ? `, up to ${pctLabel(top.pMiss)} chance of a miss` : ""} (${before})`],
    ];
  } else {
    const fa = box.filter((x) => x.obsFa), flag = box.filter((x) => x.faFlag);
    stats = [
      [`GEFS forecast for ${c.region}`, `${gefs} cells`, `with ≥ 50% of members calling ${event}`],
      ["What happened", `${fa.length} of ${gefs} didn’t`, `${event[0].toUpperCase() + event.slice(1)} never arrived in ${fa.length} of the cells GEFS called`],
      ["BustSentinel alert", `${fa.filter((x) => x.faFlag).length} of ${fa.length} flagged`,
        `${flag.length} cells flagged as likely false alarms; ${flag.filter((x) => !x.obsFa).length} of those did get ${event} (${before})`],
    ];
  }
  return (
    <Card className={cn("border-l-4", c.mode === "miss" ? "border-l-miss" : "border-l-fa")}>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <CardDescription>Case study</CardDescription>
          <CardTitle className="text-base">{c.title}</CardTitle>
        </div>
        <DirectionBadge direction={c.mode} />
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="max-w-prose text-sm text-pretty">{c.story}</p>
        <div className="grid gap-4 sm:grid-cols-3">
          {stats.map(([k, v, h]) => (
            <div key={k} className="flex flex-col gap-1 rounded-lg bg-muted/60 p-3">
              <span className="text-xs text-muted-foreground">{k}</span>
              <span className="text-xl font-semibold tabular-nums">{v}</span>
              <span className="text-xs text-pretty text-muted-foreground">{h}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export { oneIn, pct };
