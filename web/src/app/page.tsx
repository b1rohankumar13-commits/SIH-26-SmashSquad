import { Suspense } from "react";

import { Desk } from "@/components/desk/desk";

export default function Page() {
  // useSearchParams (URL-held desk state) needs a Suspense boundary.
  return (
    <Suspense>
      <Desk />
    </Suspense>
  );
}
