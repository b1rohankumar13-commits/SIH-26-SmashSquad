"use client";

import { useId } from "react";

import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatInit } from "@/lib/format";
import type { CaseStudy, Hazard, Mode } from "@/lib/types";
import type { DeskState } from "@/lib/use-desk-state";

const MODES: { value: Mode; label: string }[] = [
  { value: "both", label: "Both" },
  { value: "miss", label: "Misses" },
  { value: "false_alarm", label: "False alarms" },
];

function Field({ label, htmlFor, children }: { label: string; htmlFor?: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

export function CasePicker({ cases, value, onPick }: {
  cases: CaseStudy[]; value?: string; onPick: (c: CaseStudy | undefined) => void;
}) {
  const id = useId();
  const items = { explore: "Free exploration", ...Object.fromEntries(cases.map((c) => [c.id, c.title])) };
  return (
    <Field label="Case study" htmlFor={id}>
      <Select items={items} value={value ?? "explore"} onValueChange={(v) => onPick(cases.find((c) => c.id === v))}>
        <SelectTrigger id={id} className="w-full sm:w-[28rem]"><SelectValue /></SelectTrigger>
        <SelectContent>
          <SelectItem value="explore">Free exploration</SelectItem>
          <SelectSeparator />
          <SelectGroup>
            <SelectLabel>Validation-period cases</SelectLabel>
            {cases.map((c) => <SelectItem key={c.id} value={c.id}>{c.title}</SelectItem>)}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  );
}

export function DeskControls({ hazards, hazard, state, inits, notable, update }: {
  hazards: Hazard[];
  hazard: Hazard;
  state: DeskState;
  inits: string[];
  notable: string[];
  update: (p: Partial<DeskState>, o?: { push?: boolean }) => void;
}) {
  const ids = { hazard: useId(), init: useId(), obs: useId(), gefs: useId() };
  const others = inits.filter((i) => !notable.includes(i));
  const initItems = Object.fromEntries(inits.map((i) => [i, formatInit(i)]));
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto]">
        <Field label="Hazard" htmlFor={ids.hazard}>
          <Select
            items={Object.fromEntries(hazards.map((h) => [h.id, h.label]))}
            value={hazard.id}
            onValueChange={(v) => update({ hazard: v as string, init: undefined, mode: "both" }, { push: true })}
          >
            <SelectTrigger id={ids.hazard} className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              {hazards.map((h) => <SelectItem key={h.id} value={h.id}>{h.label}</SelectItem>)}
            </SelectContent>
          </Select>
        </Field>

        <Field label="Show">
          <ToggleGroup
            aria-label="Bust direction"
            variant="outline"
            size="sm"
            spacing={0}
            value={[hazard.kind === "grid" ? state.mode : "both"]}
            disabled={hazard.kind !== "grid"}
            onValueChange={(v) => v[0] && update({ mode: v[0] as Mode })}
            className="h-8"
          >
            {MODES.map((m) => (
              <ToggleGroupItem key={m.value} value={m.value} className="h-8 px-3">{m.label}</ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>

        <Field label="Forecast issued (00 UTC)" htmlFor={ids.init}>
          <Select items={initItems} value={state.init ?? inits[0]} onValueChange={(v) => update({ init: v as string })}>
            <SelectTrigger id={ids.init} className="w-full tabular-nums"><SelectValue /></SelectTrigger>
            <SelectContent className="max-h-80">
              {notable.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Most busts in validation</SelectLabel>
                  {notable.map((i) => <SelectItem key={`n-${i}`} value={i} className="tabular-nums">{formatInit(i)}</SelectItem>)}
                </SelectGroup>
              )}
              {notable.length > 0 && <SelectSeparator />}
              <SelectGroup>
                <SelectLabel>All dates</SelectLabel>
                {others.map((i) => <SelectItem key={i} value={i} className="tabular-nums">{formatInit(i)}</SelectItem>)}
              </SelectGroup>
            </SelectContent>
          </Select>
        </Field>

        <Field label="Lead day">
          <ToggleGroup
            aria-label="Lead day"
            variant="outline"
            size="sm"
            spacing={0}
            value={[String(state.lead)]}
            onValueChange={(v) => v[0] && update({ lead: Number(v[0]) })}
            className="h-8"
          >
            {Array.from({ length: 9 }, (_, i) => (
              <ToggleGroupItem key={i} value={String(i + 1)} aria-label={`Day ${i + 1}`} className="h-8 w-8 px-0 tabular-nums">
                {i + 1}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <label htmlFor={ids.obs} className="flex cursor-pointer items-center gap-2 text-sm">
          <Switch id={ids.obs} checked={state.observed} onCheckedChange={(v) => update({ observed: v })} />
          Show what actually happened
        </label>
        <label htmlFor={ids.gefs} className={`flex items-center gap-2 text-sm ${hazard.kind === "grid" ? "cursor-pointer" : "opacity-50"}`}>
          <Switch id={ids.gefs} checked={state.gefs} disabled={hazard.kind !== "grid"} onCheckedChange={(v) => update({ gefs: v })} />
          Show GEFS event probability
        </label>
      </div>
    </div>
  );
}
