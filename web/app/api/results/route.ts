import { NextResponse } from "next/server";
import { loadSweep } from "@/lib/results";

// Re-read ./results on every request so edits / new runs show up on reload.
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json({ datasets: await loadSweep() });
  } catch (err) {
    return NextResponse.json(
      { error: String(err instanceof Error ? err.message : err) },
      { status: 500 },
    );
  }
}
