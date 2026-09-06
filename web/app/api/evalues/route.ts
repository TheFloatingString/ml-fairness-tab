import { NextResponse } from "next/server";
import { loadEvalues } from "@/lib/results";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(await loadEvalues());
  } catch (err) {
    return NextResponse.json(
      { error: String(err instanceof Error ? err.message : err) },
      { status: 500 },
    );
  }
}
