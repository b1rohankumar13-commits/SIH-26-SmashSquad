// MapLibre 6 runs its tile worker as an ES module that Turbopack can't bundle; serve the
// published worker (+ the shared chunk it imports) from /public and point MapLibre at it.
import { copyFileSync, mkdirSync } from "node:fs";

const src = "node_modules/maplibre-gl/dist";
mkdirSync("public/maplibre", { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) copyFileSync(`${src}/${f}`, `public/maplibre/${f}`);
