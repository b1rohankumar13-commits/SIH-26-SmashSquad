import { Suspense } from "react";

import { LiveDesk } from "@/components/live/live-desk";

export default function Page() {
  // useSearchParams (URL-held state) needs a Suspense boundary.
  return (
    <Suspense>
      <LiveDesk />
    </Suspense>
  );
}
