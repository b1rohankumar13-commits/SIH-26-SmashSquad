import type { Tier } from "@/lib/types";

const fmtDate = new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const fmtShort = new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", timeZone: "UTC" });

export const parseInit = (s: string) => new Date(Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8)));
export const formatInit = (s: string) => fmtDate.format(parseInit(s));
export const validDate = (init: string, lead: number) => {
  const d = parseInit(init);
  d.setUTCDate(d.getUTCDate() + lead - 1);
  return d;
};
export const formatValid = (init: string, lead: number) => fmtDate.format(validDate(init, lead));
export const formatValidShort = (init: string, lead: number) => fmtShort.format(validDate(init, lead));

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

/** "1 in 6" for small rates, "8 in 10" for large ones. */
export function oneIn(p: number) {
  if (p <= 0) return "–";
  return p >= 0.25 ? `${Math.round(p * 10)} in 10` : `1 in ${Math.round(1 / p)}`;
}

export const article = (w: string) => (/^[aeiou]/i.test(w) ? "an" : "a");

export function tierSentence(t: Tier, direction: "miss" | "false_alarm", event: string) {
  const what = direction === "miss"
    ? `${article(event)} ${event} GEFS failed to forecast`
    : `${article(event)} ${event} GEFS forecast that didn’t happen`;
  const lift = t.lift != null && t.lift < 100 ? `${Math.round(t.lift)}× the normal rate` : "vs. well under 1% normally";
  return `${oneIn(t.precision)} flagged cells turned out to be ${what} (${lift}); catches ${pct(t.catch_rate)} of them.`;
}

export const EVENT_WORD: Record<string, string> = { rain: "extreme rain", heatwave: "heatwave", monsoon: "spell" };

export const TARGET_LABELS: Record<string, string> = {
  miss_break: "Missed break spell",
  fa_break: "False break alarm",
  miss_active: "Missed active spell",
  fa_active: "False active alarm",
};
