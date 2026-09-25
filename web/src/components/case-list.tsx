"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";

import { AppShell } from "@/components/app-shell";
import { DirectionBadge } from "@/components/desk/panels";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useCases, useHazards } from "@/lib/api";
import { formatInit, formatValid } from "@/lib/format";
import { cn } from "@/lib/utils";

export function CaseList() {
  const cases = useCases();
  const hazards = useHazards();
  const label = (id: string) => hazards.data?.find((h) => h.id === id)?.label ?? id;
  return (
    <AppShell title="Case Studies">
      <section className="flex flex-col gap-1">
        <h2 className="text-2xl font-semibold tracking-tight">Case studies</h2>
        <p className="max-w-3xl text-sm text-pretty text-muted-foreground">
          Real events from the validation years — dates the models never trained on. Open one to see what GEFS forecast, what happened, and what BustSentinel flagged.
        </p>
      </section>
      {cases.error ? (
        <p className="text-sm text-destructive">Couldn’t load case studies: {cases.error.message}</p>
      ) : !cases.data ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-48 w-full rounded-xl" />)}
        </div>
      ) : cases.data.length === 0 ? (
        <p className="text-sm text-muted-foreground">No case studies match the exported validation dates yet.</p>
      ) : (
        <ul className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {cases.data.map((c) => {
            const href = `/?${new URLSearchParams({ case: c.id, hazard: c.hazard, init: c.init, lead: String(c.lead), mode: c.mode })}`;
            return (
              <li key={c.id}>
                <Link href={href} className="group block h-full rounded-xl outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
                  <Card className={cn("h-full border-l-4 transition-[box-shadow,transform] duration-150 group-hover:-translate-y-0.5 group-hover:shadow-md",
                    c.mode === "miss" ? "border-l-miss" : "border-l-fa")}>
                    <CardHeader className="flex flex-col gap-2">
                      <div className="flex w-full items-center justify-between gap-2">
                        <CardDescription>{label(c.hazard)} · {c.region}</CardDescription>
                        <DirectionBadge direction={c.mode} />
                      </div>
                      <CardTitle className="text-base">{c.title}</CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-1 flex-col gap-3">
                      <p className="text-sm text-pretty text-muted-foreground">{c.story}</p>
                      <p className="mt-auto flex items-center justify-between text-xs text-muted-foreground tabular-nums">
                        <span>Issued {formatInit(c.init)} → valid {formatValid(c.init, c.lead)}</span>
                        <span className="flex items-center gap-1 font-medium text-foreground">
                          Open <ArrowRight aria-hidden className="size-3.5 transition-transform group-hover:translate-x-0.5" />
                        </span>
                      </p>
                    </CardContent>
                  </Card>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </AppShell>
  );
}
