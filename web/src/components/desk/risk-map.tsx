"use client";

import type { PickingInfo } from "@deck.gl/core";
import { GeoJsonLayer, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { MapboxOverlay, type MapboxOverlayProps } from "@deck.gl/mapbox";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useTheme } from "next-themes";
import { useMemo } from "react";
import { Map, useControl } from "react-map-gl/maplibre";

import { pctLabel, RAMP, rampColour, regionTone, type Cell } from "@/lib/risk";
import { areaLabel } from "@/components/desk/panels";
import type { Box, Direction, Explanation, Mode, Track, View } from "@/lib/types";

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

/** Subdivision risk for the shown lead, indexed like the GeoJSON features. */
export interface RegionLayer {
  geo: GeoJSON.FeatureCollection;
  pMiss: number[];
  pFa: number[];
  obsMiss: number[];
  obsFa: number[];
  flagMiss: number[];
  flagFa: number[];
}

type RegionFeature = GeoJSON.Feature<GeoJSON.Geometry, { id: number; name: string }>;

const regionOutcome = (m: number, f: number) =>
  m < 0 ? "No observations here" : m === 1 && f === 1 ? "GEFS missed some events and over-called others"
    : m === 1 ? "Missed: an event GEFS said no to happened" : f === 1 ? "False alarm: GEFS said yes, no event" : "Forecast held";

const outcome = (c: Cell) =>
  c.obsMiss ? "Missed: event happened, GEFS said no"
    : c.obsFa ? "False alarm: GEFS said yes, no event"
      : "Forecast held";

export function RiskMap({
  cells, mode, showObserved, box, boxTone, label, tracks, areas, view = "cells", regions,
}: {
  cells: Cell[] | null;
  mode: Mode;
  showObserved: boolean;
  box?: Box;
  boxTone?: Direction | "neutral";
  label: string;
  /** Observed low-pressure-system tracks, drawn over the risk cells. */
  tracks?: Track[];
  /** Explained areas, marked M1/F1... at their centres to match the "Why these areas?" card. */
  areas?: Explanation;
  /** Colour subdivisions by bust chance instead of cells; outlines are drawn either way. */
  view?: View;
  regions?: RegionLayer;
}) {
  const { resolvedTheme } = useTheme();
  const dark = resolvedTheme === "dark";

  const layers = useMemo(() => {
    const out = [];
    const byRegion = view === "regions" && regions;
    if (byRegion) {
      out.push(new GeoJsonLayer<{ id: number; name: string }>({
        id: "region-risk", data: regions.geo as never, pickable: true, stroked: false, filled: true,
        getFillColor: (f) => {
          const i = (f as RegionFeature).properties.id;
          const t = regions.obsMiss[i] < 0 ? null
            : regionTone(regions.pMiss[i], regions.pFa[i], regions.flagMiss[i] === 1, regions.flagFa[i] === 1, mode);
          return t ? rampColour(t.direction, t.strength) : [0, 0, 0, 0];
        },
        updateTriggers: { getFillColor: [regions, mode] },
      }));
    }
    if (cells && !byRegion) {
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
    }
    if (regions) {
      out.push(new GeoJsonLayer({
        id: "region-lines", data: regions.geo as never, filled: false, stroked: true, lineWidthUnits: "pixels",
        getLineWidth: byRegion ? 1 : 0.6, getLineColor: dark ? [255, 255, 255, byRegion ? 90 : 45] : [64, 64, 64, byRegion ? 110 : 60],
      }));
    }
    if (cells) {
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
    if (areas && cells) {
      const marks = ((mode === "both" ? ["miss", "false_alarm"] : [mode]) as Direction[])
        .flatMap((d) => areas[d].map((a, k) => ({ d, text: areaLabel(d, k), p: [a.lon, a.lat] as [number, number] })));
      out.push(new TextLayer<(typeof marks)[number]>({
        id: "area-labels", data: marks, getPosition: (m) => m.p, getText: (m) => m.text,
        getSize: 11, fontWeight: 700, fontFamily: "system-ui, sans-serif", background: true, backgroundPadding: [4, 2],
        getBackgroundColor: (m) => (m.d === "miss" ? [194, 65, 12, 235] : [29, 78, 216, 235]), getColor: [255, 255, 255, 255],
      }));
    }
    if (tracks?.length) {
      const ink: [number, number, number] = dark ? [250, 250, 250] : [23, 23, 23];
      out.push(new PathLayer<Track>({
        id: "tracks", data: tracks, getPath: (t) => t.path, getColor: [...ink, 200] as [number, number, number, number],
        getWidth: 2, widthUnits: "pixels", capRounded: true, jointRounded: true,
      }));
      out.push(new ScatterplotLayer<Track>({
        id: "track-now", data: tracks, getPosition: (t) => t.now, getRadius: 7, radiusUnits: "pixels",
        filled: true, getFillColor: dark ? [23, 23, 23, 255] : [255, 255, 255, 255],
        stroked: true, getLineColor: [...ink, 255] as [number, number, number, number], lineWidthMinPixels: 2.5,
      }));
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
  }, [cells, mode, showObserved, box, boxTone, dark, tracks, areas, view, regions]);

  const style = {
    background: dark ? "#171717" : "#ffffff", color: dark ? "#fafafa" : "#0a0a0a",
    border: "1px solid rgba(127,127,127,.25)", borderRadius: "8px", padding: "8px 10px",
    boxShadow: "0 1px 2px rgba(0,0,0,.06), 0 8px 24px rgba(0,0,0,.12)",
  };
  const tooltip = ({ object, layer }: PickingInfo) => {
    if (layer?.id === "region-risk" && regions && object) {
      const { id, name } = (object as RegionFeature).properties;
      if (regions.obsMiss[id] < 0) return { html: `<b>${name}</b><br/>Not covered for this hazard`, style };
      return {
        html: `<div style="font:12px/1.45 var(--font-sans),system-ui;font-variant-numeric:tabular-nums">
          <b>${name}</b><br/>Chance GEFS misses an event here ${pctLabel(regions.pMiss[id])} · over-calls one ${pctLabel(regions.pFa[id])}<br/>
          ${showObserved ? regionOutcome(regions.obsMiss[id], regions.obsFa[id]) : ""}</div>`,
        style,
      };
    }
    const c = object as Cell | undefined;
    if (!c || c.pMiss === undefined) return null;
    return {
      html: `<div style="font:12px/1.45 var(--font-sans),system-ui;font-variant-numeric:tabular-nums">
        <b>${c.lat.toFixed(2)}°N, ${c.lon.toFixed(2)}°E</b><br/>
        Chance of a miss ${(c.pMiss * 100).toFixed(1)}% · of a false alarm ${(c.pFa * 100).toFixed(1)}%<br/>
        GEFS event probability ${(c.ens * 100).toFixed(0)}%<br/>${outcome(c)}</div>`,
      style,
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
