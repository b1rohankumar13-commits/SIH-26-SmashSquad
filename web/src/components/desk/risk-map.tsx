"use client";

import type { PickingInfo } from "@deck.gl/core";
import { PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import { MapboxOverlay, type MapboxOverlayProps } from "@deck.gl/mapbox";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useTheme } from "next-themes";
import { useMemo } from "react";
import { Map, useControl } from "react-map-gl/maplibre";

import { RAMP, rampColour, type Cell } from "@/lib/risk";
import type { Box, Direction, Mode } from "@/lib/types";

// Worker served from /public (see scripts/copy-maplibre-worker.mjs).
setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");

const BASEMAP = {
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};
// Model grid 5-38N, 65-100E; the view hugs it.
const VIEW = { longitude: 82.5, latitude: 21.8, zoom: 3.55, minZoom: 3.2, maxZoom: 8, pitch: 0, bearing: 0 };
const H = 0.25;

/** deck.gl drawn as a MapLibre control - the pattern deck.gl recommends for MapLibre. */
function DeckOverlay(props: MapboxOverlayProps) {
  const overlay = useControl<MapboxOverlay>(() => new MapboxOverlay(props));
  overlay.setProps(props);
  return null;
}

const square = (c: Cell) => [
  [c.lon - H, c.lat - H], [c.lon + H, c.lat - H], [c.lon + H, c.lat + H], [c.lon - H, c.lat + H],
];

const outcome = (c: Cell) =>
  c.obsMiss ? "Missed: event happened, GEFS said no"
    : c.obsFa ? "False alarm: GEFS said yes, no event"
      : "Forecast held";

export function RiskMap({
  cells, mode, showObserved, showGefs, box, boxTone, label,
}: {
  cells: Cell[] | null;
  mode: Mode;
  showObserved: boolean;
  showGefs: boolean;
  box?: Box;
  boxTone?: Direction | "neutral";
  label: string;
}) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";

  const layers = useMemo(() => {
    const out = [];
    if (cells) {
      if (showGefs) {
        out.push(new PolygonLayer<Cell>({
          id: "gefs", data: cells.filter((c) => c.ens > 0), getPolygon: square, stroked: false,
          getFillColor: (c) => [20, 184, 166, Math.round(40 + 150 * c.ens)],
        }));
      }
      out.push(new PolygonLayer<Cell>({
        id: "risk", data: cells.filter((c) => c.direction !== null), getPolygon: square, pickable: true,
        getFillColor: (c) => rampColour(c.direction!, c.strength),
        getLineColor: dark ? [255, 255, 255, 40] : [255, 255, 255, 90], lineWidthMinPixels: 0.5,
        updateTriggers: { getFillColor: [mode] },
      }));
      // Invisible pickable layer so every observed cell has a tooltip, flagged or not.
      out.push(new PolygonLayer<Cell>({
        id: "hover", data: cells, getPolygon: square, pickable: true, stroked: false, getFillColor: [0, 0, 0, 0],
      }));
      if (showObserved) {
        for (const d of (mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[]) {
          out.push(new ScatterplotLayer<Cell>({
            id: `obs-${d}`, data: cells.filter((c) => (d === "miss" ? c.obsMiss : c.obsFa)),
            getPosition: (c) => [c.lon, c.lat], getRadius: 11000, radiusMinPixels: 2.5, radiusMaxPixels: 8,
            filled: false, stroked: true, lineWidthMinPixels: 1.5,
            // Light rings on the dark basemap; near-black on light (visible on fills and the pale basemap).
            getLineColor: [...(dark ? RAMP[d].ringDark : RAMP[d].ringLight)] as [number, number, number, number],
          }));
        }
      }
    }
    if (box) {
      const tone = boxTone === "miss" ? [234, 88, 12] : boxTone === "false_alarm" ? [37, 99, 235] : [100, 116, 139];
      out.push(new PolygonLayer({
        id: "box", data: [box],
        getPolygon: (b: Box) => [[b.west, b.south], [b.east, b.south], [b.east, b.north], [b.west, b.north]],
        getFillColor: cells ? [0, 0, 0, 0] : [...tone, 70] as [number, number, number, number],
        getLineColor: [...tone, 230] as [number, number, number], lineWidthMinPixels: 2,
      }));
    }
    return out;
  }, [cells, mode, showObserved, showGefs, box, boxTone, dark]);

  const tooltip = ({ object }: PickingInfo) => {
    const c = object as Cell | undefined;
    if (!c || c.pMiss === undefined) return null;
    return {
      html: `<div style="font:12px/1.45 var(--font-sans),system-ui;font-variant-numeric:tabular-nums">
        <b>${c.lat.toFixed(2)}°N, ${c.lon.toFixed(2)}°E</b><br/>
        Chance of a miss ${(c.pMiss * 100).toFixed(1)}% · of a false alarm ${(c.pFa * 100).toFixed(1)}%<br/>
        GEFS event probability ${(c.ens * 100).toFixed(0)}%<br/>${outcome(c)}</div>`,
      style: { background: dark ? "#171717" : "#ffffff", color: dark ? "#fafafa" : "#0a0a0a",
        border: "1px solid rgba(127,127,127,.25)", borderRadius: "8px", padding: "8px 10px",
        boxShadow: "0 1px 2px rgba(0,0,0,.06), 0 8px 24px rgba(0,0,0,.12)" },
    };
  };

  return (
    <div role="img" aria-label={label} className="relative h-[min(68vh,560px)] min-h-[340px] w-full overflow-hidden rounded-lg">
      <Map
        initialViewState={VIEW}
        minZoom={VIEW.minZoom}
        maxZoom={VIEW.maxZoom}
        dragRotate={false}
        touchPitch={false}
        mapStyle={dark ? BASEMAP.dark : BASEMAP.light}
        attributionControl={{ compact: true }}
        style={{ width: "100%", height: "100%" }}
      >
        <DeckOverlay layers={layers} getTooltip={tooltip} />
      </Map>
    </div>
  );
}
