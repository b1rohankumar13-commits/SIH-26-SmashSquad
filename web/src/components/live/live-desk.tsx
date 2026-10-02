"use client";

import { AlertCircle, CheckCircle2, Download, FileDown, Loader2 } from "lucide-react";
import dynamic from "next/dynamic";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useId, useMemo, useState, useTransition } from "react";

import { AppShell } from "@/components/app-shell";
import { ExplainCard, MapLegend } from "@/components/desk/panels";
import { ConfidenceCard, ConfidenceLegend, RegionReasonsCard } from "@/components/desk/confidence";
import { RegionCard } from "@/components/desk/region-card";
import type { RegionLayer } from "@/components/desk/risk-map";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  setLiveAuto, startLiveRun, useHazards, useLiveAuto, useLiveExplain, useLiveGrid, useLiveJob, useLiveRegions, useLiveRuns, useSubdivisions,
} from "@/lib/api";
import { EVENT_WORD, formatInit, formatValid } from "@/lib/format";
import { atIntensity, classify, pctLabel, scaleFor, SHOW_FROM, type Scale, type Thresholds } from "@/lib/risk";
import type { Direction, Hazard, LiveRun, LiveSource, Mode, View } from "@/lib/types";

const RiskMap = dynamic(() => import("@/components/desk/risk-map").then((m) => m.RiskMap), {
  ssr: false, loading: () => <Skeleton className="h-[min(68vh,560px)] min-h-[340px] w-full" />,
});

const SOURCE: Record<LiveSource, { short: string; note: string }> = {
  gefs: { short: "GEFS", note: "NOAA, published ~5 h after 00 UTC" },
  neps: { short: "NEPS", note: "NCMRWF via TIGGE, 48 h delay" },
};
const MODES: { value: Mode; label: string }[] = [
  { value: "both", label: "Both" }, { value: "miss", label: "Misses" }, { value: "false_alarm", label: "False alarms" },
];
const DIRECTION_WORD: Record<Direction, string> = { miss: "misses", false_alarm: "false alarms" };
const OTHER: Record<LiveSource, LiveSource> = { gefs: "neps", neps: "gefs" };

/** URL-held state, as on the desk. */
function useLiveState() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [, startTransition] = useTransition();
  const lead = Number(params.get("lead") ?? 1);
  const mode = params.get("mode") as Mode | null;
  const state = {
    source: (params.get("source") === "neps" ? "neps" : "gefs") as LiveSource,
    init: params.get("init") ?? undefined,
    hazard: params.get("hazard") ?? undefined,
    lead: Number.isInteger(lead) && lead >= 1 && lead <= 9 ? lead : 1,
    mode: mode && MODES.some((m) => m.value === mode) ? mode : ("both" as Mode),
    view: ((["regions", "confidence"] as const).find((v) => v === params.get("view")) ?? "cells") as View,
    compare: params.get("compare") === "1",
    observed: params.get("observed") !== "0",
  };
  const update = useCallback((patch: Partial<typeof state>) => {
    const next = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(patch)) {
      const def = { lead: 1, mode: "both", view: "cells", compare: false, observed: true, source: "gefs" }[k];
      if (v === undefined || v === def) next.delete(k);
      else next.set(k, typeof v === "boolean" ? (v ? "1" : "0") : String(v));
    }
    startTransition(() => router.replace(`${pathname}?${next.toString()}`, { scroll: false }));
  }, [params, pathname, router]);
  return { state, update };
}

function Field({ label, htmlFor, children }: { label: string; htmlFor?: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

/** Automatic daily GEFS run (in the API process): fetches and analyses each new run as it is posted. */
function AutoSwitch({ onNewRun }: { onNewRun: () => void }) {
  const auto = useLiveAuto();
  const a = auto.data;
  const job = useLiveJob(a?.last_job ?? undefined);
  useEffect(() => {
    if (job.data?.status === "done") onNewRun();
  }, [job.data?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  const toggle = async (v: boolean) => auto.mutate(await setLiveAuto(v), { revalidate: false });
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
      <label className="flex items-center gap-2">
        <Switch checked={!!a?.enabled} disabled={!a} onCheckedChange={toggle} />
        Update automatically every day (GEFS)
      </label>
      {a?.enabled && (
        <span className="text-xs text-muted-foreground tabular-nums">
          checks every {a.every_minutes} min
          {a.last_init && ` · last fetched ${formatInit(a.last_init)}${job.data?.status === "running" ? " (running…)" : ""}`}
        </span>
      )}
    </div>
  );
}

/** Fetch the newest forecast for a source and run the models on it, with live progress. */
function FetchCard({ source, onDone }: { source: LiveSource; onDone: (init: string) => void }) {
  const [job, setJob] = useState<string>();
  const [error, setError] = useState<string>();
  // Empty = newest available. A past date runs on observed inputs so it can be validated.
  const [date, setDate] = useState("");
  // Demo mode: the download is simulated and the forecast comes from disk (no network / queue risk).
  const [demo, setDemo] = useState(true);
  const status = useLiveJob(job);
  const running = status.data?.status === "running" || (job && !status.data);
  useEffect(() => {
    if (status.data?.status === "done" && status.data.init) onDone(status.data.init);
  }, [status.data?.status, status.data?.init]); // eslint-disable-line react-hooks/exhaustive-deps
  const go = async () => {
    setError(undefined);
    try {
      const init = date ? date.replaceAll("-", "") : undefined;
      setJob((await startLiveRun({ source, init, fetch: true, realtime: !init, demo })).job);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const last = status.data?.log.at(-1);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={go} disabled={!!running}>
          {running ? <Loader2 aria-hidden className="animate-spin" /> : <Download aria-hidden />}
          Fetch {date ? "" : "newest "}{SOURCE[source].short} &amp; analyse
        </Button>
        <Input type="date" aria-label="Forecast date (empty = newest)" value={date} onChange={(e) => setDate(e.target.value)}
          className="h-8 w-40 tabular-nums" />
        <span className="text-xs text-muted-foreground">{SOURCE[source].note}</span>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={demo} onCheckedChange={setDemo} size="sm" />
          Demo mode (saved data)
        </label>
      </div>
      {(running || status.data) && (
        <p aria-live="polite" className="flex items-center gap-2 text-xs text-muted-foreground">
          {status.data?.status === "done" ? <CheckCircle2 aria-hidden className="size-3.5 text-foreground" />
            : status.data?.status === "failed" ? <AlertCircle aria-hidden className="size-3.5 text-destructive" /> : null}
          <span className="truncate font-mono">{status.data?.status === "failed" ? `Failed: ${last}` : last ?? "Starting…"}</span>
        </p>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

function ValidationCard({ run, hazard, mode }: { run: LiveRun; hazard: string; mode: Mode }) {
  const v = run.hazards[hazard]?.validation;
  if (!v) return null;
  const dirs = (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[];
  return (
    <Card size="sm">
      <CardHeader><CardTitle>What actually happened</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-1 text-sm tabular-nums">
        {dirs.map((d) => (
          <p key={d}>
            <b className="font-medium">{v[d].busts}</b> {DIRECTION_WORD[d]} over Days 1–9
            {v[d].busts > 0 && <>; the coloured area caught <b className="font-medium">{v[d].caught}</b></>}
            <span className="text-muted-foreground"> ({v[d].flagged} cells coloured)</span>
          </p>
        ))}
      </CardContent>
    </Card>
  );
}

/** One source's map for the chosen hazard/day. */
function LiveMap({ run, hazard, lead, mode, view, observed, title }: {
  run: LiveRun; hazard: Hazard; lead: number; mode: Mode; view: View; observed: boolean; title?: string;
}) {
  const grid = useLiveGrid(run.source, run.init, hazard.id, lead);
  const regions = useLiveRegions(run.source, run.init, view === "regions" ? hazard.id : undefined);
  const explain = useLiveExplain(view === "cells" ? run.source : undefined, run.init, hazard.id, lead);
  const geo = useSubdivisions(true);
  const thresholds = useMemo<Thresholds>(() => ({
    miss: atIntensity(SHOW_FROM, scaleFor(hazard.tiers.miss)), false_alarm: atIntensity(SHOW_FROM, scaleFor(hazard.tiers.false_alarm)),
  }), [hazard]);
  const cells = useMemo(() => (grid.data ? classify(grid.data, hazard.tiers, mode, thresholds) : null), [grid.data, hazard, mode, thresholds]);
  const L = lead - 1;
  const regionLayer = useMemo<RegionLayer | undefined>(() => (geo.data && regions.data ? {
    geo: geo.data, pMiss: regions.data.p_miss[L], pFa: regions.data.p_fa[L], obsMiss: regions.data.obs_miss[L],
    obsFa: regions.data.obs_fa[L], flagMiss: regions.data.flag_miss[L], flagFa: regions.data.flag_fa[L],
    cuts: hazard.regions?.confidence_cuts,
  } : geo.data ? { geo: geo.data, pMiss: [], pFa: [], obsMiss: [], obsFa: [], flagMiss: [], flagFa: [] } : undefined),
  [geo.data, regions.data, L, hazard]);
  const shift = run.hazards[hazard.id]?.valid_shift;
  const validated = !!run.hazards[hazard.id]?.validated;
  return (
    <Card className="gap-3">
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <div className="flex flex-col gap-1">
          <CardTitle>{title ?? `${hazard.label} bust risk · Day ${lead}`}</CardTitle>
          <CardDescription>Valid {formatValid(run.init, lead, shift)} · {SOURCE[run.source].short} issued {formatInit(run.init)} 00 UTC</CardDescription>
        </div>
        <Badge variant="outline">{cells ? `${cells.filter((c) => c.missFlag).length} miss · ${cells.filter((c) => c.faFlag).length} FA cells` : "…"}</Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <RiskMap cells={cells} mode={mode} showObserved={observed && validated} view={view}
          regions={regionLayer}
          areas={view === "cells" ? explain.data : undefined}
          label={`${hazard.label} bust-risk map, ${SOURCE[run.source].short}, valid ${formatValid(run.init, lead, shift)}`} />
        {cells && !cells.some((c) => c.missFlag || c.faFlag) && (
          <p className="text-sm text-muted-foreground">
            Nothing crosses the alert threshold on Day {lead}: the highest chance of a bust anywhere is{" "}
            {pctLabel(Math.max(...cells.map((c) => Math.max(c.pMiss, c.pFa))))}, so {SOURCE[run.source].short} is expected to hold.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export function LiveDesk() {
  const { state, update } = useLiveState();
  const runs = useLiveRuns();
  const hazards = useHazards();
  const ids = { run: useId(), hazard: useId(), obs: useId() };

  const mine = (runs.data ?? []).filter((r) => r.source === state.source);
  const run = mine.find((r) => r.init === state.init) ?? mine[0];
  const other = run && runs.data?.find((r) => r.source === OTHER[run.source] && r.init === run.init);
  const okHazards = run ? Object.entries(run.hazards).filter(([, v]) => v.status === "ok").map(([k]) => k) : [];
  const hazard = hazards.data?.find((h) => h.id === (state.hazard && okHazards.includes(state.hazard) ? state.hazard : okHazards[0]));
  const compare = state.compare && !!other && other.hazards[hazard?.id ?? ""]?.status === "ok";
  const validated = !!(run && hazard && run.hazards[hazard.id]?.validated);
  const scales = useMemo<Record<Direction, Scale> | null>(() => (hazard
    ? { miss: scaleFor(hazard.tiers.miss), false_alarm: scaleFor(hazard.tiers.false_alarm) } : null), [hazard]);
  const regions = useLiveRegions(run?.source, run?.init, hazard?.id);
  const subs = run && hazard ? run.hazards[hazard.id]?.substitutions ?? [] : [];
  const explain = useLiveExplain(run?.source, run?.init, hazard?.id, state.lead);
  const explainOther = useLiveExplain(compare ? other?.source : undefined, other?.init, hazard?.id, state.lead);

  return (
    <AppShell title="Live Forecast">
      <section aria-labelledby="live-title" className="flex flex-col gap-1">
        <h2 id="live-title" className="text-2xl font-semibold tracking-tight">Live forecast analysis</h2>
        <p className="max-w-3xl text-sm text-pretty text-muted-foreground">
          Pull the newest ensemble forecast, see where it is likely to bust, and compare GEFS with NEPS.
        </p>
      </section>

      <Card size="sm">
        <CardContent className="flex flex-col gap-4">
          <div className="grid gap-4 lg:grid-cols-[auto_minmax(0,1fr)]">
            <Field label="Forecast system">
              <ToggleGroup aria-label="Forecast system" variant="outline" size="sm" spacing={0} value={[state.source]}
                onValueChange={(v) => v[0] && update({ source: v[0] as LiveSource, init: undefined })} className="h-8">
                {(["gefs", "neps"] as LiveSource[]).map((s) => (
                  <ToggleGroupItem key={s} value={s} className="h-8 px-3">{SOURCE[s].short}</ToggleGroupItem>
                ))}
              </ToggleGroup>
            </Field>
            <FetchCard source={state.source} onDone={(init) => { runs.mutate(); update({ init }); }} />
          </div>
          <AutoSwitch onNewRun={() => runs.mutate()} />

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_auto]">
            <Field label="Forecast issued (00 UTC)" htmlFor={ids.run}>
              <Select items={Object.fromEntries(mine.map((r) => [r.init, formatInit(r.init)]))} value={run?.init ?? null}
                onValueChange={(v) => update({ init: v as string })} disabled={!mine.length}>
                <SelectTrigger id={ids.run} className="w-full tabular-nums"><SelectValue placeholder="No runs yet" /></SelectTrigger>
                <SelectContent>
                  {mine.map((r) => <SelectItem key={r.init} value={r.init} className="tabular-nums">{formatInit(r.init)}</SelectItem>)}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Hazard" htmlFor={ids.hazard}>
              <Select items={Object.fromEntries((hazards.data ?? []).map((h) => [h.id, h.label]))} value={hazard?.id ?? null}
                onValueChange={(v) => update({ hazard: v as string })} disabled={!okHazards.length}>
                <SelectTrigger id={ids.hazard} className="w-full"><SelectValue placeholder="—" /></SelectTrigger>
                <SelectContent>
                  {(hazards.data ?? []).filter((h) => run?.hazards[h.id]).map((h) => (
                    <SelectItem key={h.id} value={h.id} disabled={run?.hazards[h.id].status !== "ok"}>
                      {h.label}{run?.hazards[h.id].status === "out_of_season" ? " (out of season)" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Show">
              <ToggleGroup aria-label="Bust direction" variant="outline" size="sm" spacing={0} value={[state.mode]}
                onValueChange={(v) => v[0] && update({ mode: v[0] as Mode })} className="h-8">
                {MODES.map((m) => <ToggleGroupItem key={m.value} value={m.value} className="h-8 px-3">{m.label}</ToggleGroupItem>)}
              </ToggleGroup>
            </Field>
            <Field label="Lead day">
              <ToggleGroup aria-label="Lead day" variant="outline" size="sm" spacing={0} value={[String(state.lead)]}
                onValueChange={(v) => v[0] && update({ lead: Number(v[0]) })} className="h-8">
                {Array.from({ length: 9 }, (_, i) => (
                  <ToggleGroupItem key={i} value={String(i + 1)} aria-label={`Day ${i + 1}`} className="h-8 w-8 px-0 tabular-nums">{i + 1}</ToggleGroupItem>
                ))}
              </ToggleGroup>
            </Field>
          </div>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <ToggleGroup aria-label="Map by" variant="outline" size="sm" spacing={0} value={[state.view]}
              onValueChange={(v) => v[0] && update({ view: v[0] as View })} className="h-8">
              <ToggleGroupItem value="cells" className="h-8 px-3">Grid cells</ToggleGroupItem>
              <ToggleGroupItem value="regions" className="h-8 px-3">Subdivisions</ToggleGroupItem>
              <ToggleGroupItem value="confidence" className="h-8 px-3">Confidence</ToggleGroupItem>
            </ToggleGroup>
            {run && hazard && (
              <a href={`/api/bulletin/${hazard.id}.csv?init=${run.init}&source=${run.source}`} download
            className={buttonVariants({ variant: "outline", size: "sm" })}>
            <FileDown aria-hidden />Download bulletin (CSV)
          </a>
            )}
            <label className="flex items-center gap-2 text-sm">
              <Switch checked={state.compare && !!other} disabled={!other} onCheckedChange={(v) => update({ compare: v })} />
              Compare with {SOURCE[OTHER[state.source]].short}{!other && run ? " (no run for this date)" : ""}
            </label>
            <label htmlFor={ids.obs} className="flex items-center gap-2 text-sm">
              <Switch id={ids.obs} checked={state.observed && validated} disabled={!validated} onCheckedChange={(v) => update({ observed: v })} />
              Validate against what happened{run && hazard && !validated ? " (not observed yet)" : ""}
            </label>
          </div>
        </CardContent>
      </Card>

      {runs.error || hazards.error ? (
        <Card><CardContent className="text-sm text-destructive">{(runs.error ?? hazards.error).message}</CardContent></Card>
      ) : !runs.data || !hazards.data ? <Skeleton className="h-96 w-full" /> : !run ? (
        <Card><CardContent className="text-sm text-muted-foreground">
          No {SOURCE[state.source].short} forecasts analysed yet. Fetch the newest one above.
        </CardContent></Card>
      ) : !hazard ? (
        <Card><CardContent className="text-sm text-muted-foreground">No hazard could run for this forecast.</CardContent></Card>
      ) : (
        <>
          <div className={compare ? "grid gap-4 xl:grid-cols-2" : "grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]"}>
            <LiveMap run={run} hazard={hazard} lead={state.lead} mode={state.mode} view={state.view} observed={state.observed}
              title={compare ? `${SOURCE[run.source].short} · ${hazard.label} · Day ${state.lead}` : undefined} />
            {compare && other ? (
              <LiveMap run={other} hazard={hazard} lead={state.lead} mode={state.mode} view={state.view} observed={state.observed}
                title={`${SOURCE[other.source].short} · ${hazard.label} · Day ${state.lead}`} />
            ) : (
              <div className="flex flex-col gap-4">
                {state.view === "cells" ? <ExplainCard data={explain.data} mode={state.mode} />
                  : <RegionReasonsCard data={regions.data} explain={explain.data} lead={state.lead} mode={state.mode}
                      cuts={state.view === "confidence" ? hazard.regions?.confidence_cuts : undefined} />}
                {state.observed && validated && <ValidationCard run={run} hazard={hazard.id} mode={state.mode} />}
                {state.view === "confidence" ? <Card size="sm"><CardHeader><CardTitle>How to read the map</CardTitle></CardHeader>
                  <CardContent><ConfidenceLegend /></CardContent></Card> : scales && <MapLegend scales={scales} mode={state.mode} event={EVENT_WORD[hazard.id] ?? "event"}
                  showObserved={state.observed && validated} byRegion={state.view === "regions"} />}
                {subs.length > 0 && (
                  <Card size="sm">
                    <CardHeader><CardTitle>Real-time inputs</CardTitle></CardHeader>
                    <CardContent>
                      <ul className="flex list-disc flex-col gap-1 pl-4 text-xs text-muted-foreground">
                        {subs.map((s) => <li key={s}>{s.replaceAll("->", "→")}</li>)}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </div>
          {compare && other && (
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="flex flex-col gap-4">
                {state.observed && <ValidationCard run={run} hazard={hazard.id} mode={state.mode} />}
                <ExplainCard data={explain.data} mode={state.mode} />
              </div>
              <div className="flex flex-col gap-4">
                {state.observed && <ValidationCard run={other} hazard={hazard.id} mode={state.mode} />}
                <ExplainCard data={explainOther.data} mode={state.mode} />
              </div>
            </div>
          )}
          {state.view === "confidence"
            ? <ConfidenceCard data={regions.data} cuts={hazard.regions?.confidence_cuts} lead={state.lead}
                showObserved={state.observed && validated} onLead={(lead) => update({ lead })} />
            : <RegionCard hazard={hazard} data={regions.data} mode={state.mode} lead={state.lead} showSkill={false}
                showObserved={state.observed && validated} onLead={(lead) => update({ lead })} />}
        </>
      )}
    </AppShell>
  );
}
