import { NextResponse } from "next/server";
import { loadCorrection } from "@/lib/results";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json({ datasets: await loadCorrection() });
  } catch (err) {
    return NextResponse.json(
      { error: String(err instanceof Error ? err.message : err) },
      { status: 500 },
    );
  }
}
