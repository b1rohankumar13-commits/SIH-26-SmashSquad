import type { Metadata } from "next";
import { Suspense } from "react";

import { CaseList } from "@/components/case-list";

export const metadata: Metadata = { title: "Case Studies" };

export default function CasesPage() {
  return (
    <Suspense>
      <CaseList />
    </Suspense>
  );
}
