import { NextResponse } from "next/server";
import { loadDriftCorrelation } from "@/lib/results";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json({ models: await loadDriftCorrelation() });
  } catch (err) {
    return NextResponse.json(
      { error: String(err instanceof Error ? err.message : err) },
      { status: 500 },
    );
  }
}
