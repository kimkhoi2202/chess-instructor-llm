// Typed client for the FastAPI coach backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type Tier = "beginner" | "intermediate" | "advanced";

export interface Example {
  label: string;
  fen: string;
  tier: Tier;
  student_move: string | null;
}

export interface EngineMove {
  san: string;
  uci: string;
  cp: number;
  pv: string[];
}

export interface StudentInfo {
  san: string;
  uci: string;
  cp_loss: number;
  severity: string;
}

export interface EngineBlock {
  best_san: string;
  best_cp: number;
  sound_pool: EngineMove[];
  student_move: StudentInfo | null;
}

export interface MaiaInfo {
  san: string;
  uci: string;
  policy: number;
}

export interface CoachMeta {
  model: string;
  tuned: boolean;
  notes: string[];
  /** How many generations the faithfulness gate needed (1 = clean first try). */
  attempts: number;
  /** True when every attempt failed and a verified, engine-derived reply was used. */
  verified_fallback: boolean;
}

export interface CoachResponse {
  recommended_move_san: string;
  recommended_move_uci: string;
  coaching: string;
  takeaway: string;
  concepts_used: string[];
  side_to_move: "white" | "black";
  engine: EngineBlock;
  maia: MaiaInfo[];
  meta: CoachMeta;
}

export interface CoachRequest {
  fen: string;
  tier: Tier;
  student_move?: string | null;
}

/** A pre-computed library entry: a real dataset position + the tuned model's
 *  cached coaching, so the gallery renders instantly without a model call. */
export interface LibraryEntry {
  id: string;
  label: string;
  fen: string;
  tier: Tier;
  phase: string;
  severity: string | null;
  student_move: string | null;
  coach: CoachResponse;
}

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail) && data.detail[0]?.msg) return data.detail[0].msg;
  } catch {
    /* fall through */
  }
  return `Request failed (${res.status})`;
}

export async function getExamples(signal?: AbortSignal): Promise<Example[]> {
  const res = await fetch(`${API_BASE}/api/examples`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getHealth(
  signal?: AbortSignal,
): Promise<{ status: string; model: string; tuned: boolean; ready: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/** The pre-generated position library (static asset served by Next from public/). */
export async function getLibrary(signal?: AbortSignal): Promise<LibraryEntry[]> {
  const res = await fetch("/library.json", { signal, cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

export async function postCoach(req: CoachRequest, signal?: AbortSignal): Promise<CoachResponse> {
  const res = await fetch(`${API_BASE}/api/coach`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}
