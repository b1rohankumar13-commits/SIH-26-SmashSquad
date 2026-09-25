"use client";

import { AlertCircle } from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useMemo } from "react";

import { AppShell } from "@/components/app-shell";
import { CasePicker, DeskControls } from "@/components/desk/controls";
import { MonsoonView } from "@/components/desk/monsoon-view";
import { CaseCard, GridMetrics, LeadChart, MapLegend, MetricsSkeleton, ModelFacts } from "@/components/desk/panels";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useCases, useGrid, useHazards, useInits, useLeads, useMonsoon } from "@/lib/api";
import { EVENT_WORD, formatInit, formatValid, oneIn } from "@/lib/format";
import { atIntensity, classify, scaleFor, SHOW_FROM, type Scale, type Thresholds } from "@/lib/risk";
import type { CaseStudy } from "@/lib/types";
import { useDeskState } from "@/lib/use-desk-state";

const RiskMap = dynamic(() => import("@/components/desk/risk-map").then((m) => m.RiskMap), {
  ssr: false, loading: () => <Skeleton className="h-[min(68vh,560px)] min-h-[340px] w-full" />,
});

const MODE_LABEL = { both: "Both directions", miss: "Misses", false_alarm: "False alarms" } as const;

function ErrorState({ message }: { message: string }) {
  return (
    <Card>
      <CardContent className="flex items-start gap-3">
        <AlertCircle aria-hidden className="mt-0.5 size-5 text-destructive" />
        <div className="flex flex-col gap-1 text-sm">
          <p className="font-medium">Couldn’t load forecast data</p>
          <p className="text-muted-foreground">{message}</p>
          <p className="text-muted-foreground">
            Start the API with <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">uvicorn api.main:app --port 8000</code> and refresh.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export function Desk() {
  const { state, update, pending } = useDeskState();
  const hazards = useHazards();
  const cases = useCases();
  const hazard = hazards.data?.find((h) => h.id === state.hazard) ?? hazards.data?.[0];
  const inits = useInits(hazard?.id);
  const notable = inits.data?.notable ?? [];
  const init = state.init && inits.data?.inits.includes(state.init) ? state.init : notable[0] ?? inits.data?.inits[0];
  const grid = useGrid(hazard?.kind === "grid" ? hazard.id : undefined, init, state.lead);
  const leads = useLeads(hazard?.kind === "grid" ? hazard.id : undefined, init);
  const monsoon = useMonsoon(hazard?.kind === "regional" ? init : undefined);

  const caseStudy = cases.data?.find((c) => c.id === state.caseId);
  const activeCase = caseStudy && caseStudy.hazard === hazard?.id && caseStudy.init === init && caseStudy.lead === state.lead
    ? caseStudy : undefined;
  const mode = hazard?.kind === "grid" ? state.mode : "both";
  const scales = useMemo<Record<"miss" | "false_alarm", Scale> | null>(() => (hazard?.kind === "grid"
    ? { miss: scaleFor(hazard.tiers.miss), false_alarm: scaleFor(hazard.tiers.false_alarm) } : null), [hazard]);
  // Counts and case studies use the same cut-off as the map: a counted cell is a coloured cell.
  const thresholds = useMemo<Thresholds | null>(() => (scales
    ? { miss: atIntensity(SHOW_FROM, scales.miss), false_alarm: atIntensity(SHOW_FROM, scales.false_alarm) } : null), [scales]);
  const cells = useMemo(() => (grid.data && hazard && thresholds ? classify(grid.data, hazard.tiers, mode, thresholds) : null),
    [grid.data, hazard, mode, thresholds]);

  useEffect(() => {
    document.title = hazard && init ? `${hazard.label} · ${formatInit(init)} · Day ${state.lead} · BustSentinel` : "Forecast Desk · BustSentinel";
  }, [hazard, init, state.lead]);

  const pickCase = (c: CaseStudy | undefined) =>
    c ? update({ caseId: c.id, hazard: c.hazard, init: c.init, lead: c.lead, mode: c.mode, observed: true }, { push: true })
      : update({ caseId: undefined });

  const event = EVENT_WORD[hazard?.id ?? "rain"] ?? "event";
  const error = hazards.error ?? inits.error ?? grid.error ?? monsoon.error;

  return (
    <AppShell title="Forecast Desk" actions={pending ? <Badge variant="outline" aria-live="polite">Updating…</Badge> : null}>
      <section aria-labelledby="desk-title" className="flex flex-col gap-1">
        <h2 id="desk-title" className="text-2xl font-semibold tracking-tight">Where will the forecast bust — and which way?</h2>
        <p className="max-w-3xl text-sm text-pretty text-muted-foreground">
          Per-cell risk that the GEFS ensemble is confidently wrong: missing an event it should have called, or calling one that never comes.
        </p>
      </section>

      {error ? <ErrorState message={error.message} /> : !hazards.data || !hazard || !inits.data ? (
        <div className="flex flex-col gap-4"><Skeleton className="h-16 w-full" /><MetricsSkeleton /></div>
      ) : (
        <>
          <Card size="sm">
            <CardContent className="flex flex-col gap-4">
              {cases.data && cases.data.length > 0 && <CasePicker cases={cases.data} value={state.caseId} onPick={pickCase} />}
              <DeskControls hazards={hazards.data} hazard={hazard} state={{ ...state, init }}
                inits={inits.data.inits} notable={notable} update={update} />
            </CardContent>
          </Card>

          {hazard.kind === "regional" ? (
            <MonsoonView hazard={hazard} data={monsoon.data} lead={state.lead} showObserved={state.observed} />
          ) : (
            <>
              {cells && thresholds ? <GridMetrics cells={cells} thresholds={thresholds} event={event} validLabel={formatValid(init!, state.lead)} /> : <MetricsSkeleton />}
              {activeCase && cells && <CaseCard c={activeCase} cells={cells} event={event} lead={state.lead} />}
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
                <Card className="gap-3">
                  <CardHeader className="flex flex-row items-center justify-between gap-2">
                    <div className="flex flex-col gap-1">
                      <CardTitle>{hazard.label} bust risk · Day {state.lead}</CardTitle>
                      <CardDescription>Valid {formatValid(init!, state.lead)} · issued {formatInit(init!)} 00 UTC</CardDescription>
                    </div>
                    <Badge variant="outline">{MODE_LABEL[mode]}</Badge>
                  </CardHeader>
                  <CardContent>
                    <RiskMap
                      cells={cells} mode={mode} showObserved={state.observed} showGefs={state.gefs}
                      box={activeCase?.box} boxTone={activeCase?.mode}
                      label={cells ? `${hazard.label} bust-risk map for ${formatValid(init!, state.lead)}: ${cells.filter((c) => c.missFlag).length} cells flagged for a likely miss, ${cells.filter((c) => c.faFlag).length} for a likely false alarm.` : "Loading map"}
                    />
                  </CardContent>
                </Card>
                <div className="flex flex-col gap-4">
                  {scales && <MapLegend scales={scales} mode={mode} event={event} showObserved={state.observed} showGefs={state.gefs} />}
                  <ModelFacts hazard={hazard} falseAlarmNote={hazard.id === "rain"
                    ? `False alarms are largely visible in GEFS itself: when most members call ${event}, it doesn’t happen about ${oneIn(hazard.tiers.false_alarm.high.precision)} times. The model’s value is mainly on misses — events GEFS gave no warning of.`
                    : undefined} />
                </div>
              </div>
              <LeadChart rows={leads.data} mode={mode} showObserved={state.observed} lead={state.lead} init={init!} />
            </>
          )}
        </>
      )}
    </AppShell>
  );
}
