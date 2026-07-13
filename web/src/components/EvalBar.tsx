"use client";

import type { CSSProperties } from "react";

type Side = "white" | "black";

// Centipawns from the coaching engine facts are stored side-to-move POV (see
// src/engine/stockfish_engine.py); we convert to WHITE POV for the bar. Mate is
// projected onto a large |cp| (>= MATE), matching the app's other eval readouts.
const MATE = 20000;

// Lichess-style win% for White from a WHITE-POV centipawn eval, clamped so the
// fill is never exactly empty/full (a sliver of the losing side always remains).
function whiteWinPct(cpWhite: number): number {
  if (cpWhite >= MATE) return 100;
  if (cpWhite <= -MATE) return 0;
  const pct = 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * cpWhite)) - 1);
  return Math.max(3, Math.min(97, pct));
}

// Magnitude shown at the leader's end (Lichess style): "3.6", "0.2", or "#".
function magnitudeLabel(cpWhite: number): string {
  if (Math.abs(cpWhite) >= MATE) return "#";
  return (Math.abs(cpWhite) / 100).toFixed(1);
}

interface EvalBarProps {
  /** Best-move centipawns from the engine facts (side-to-move POV), or null when
   *  no eval is available for the board on screen (renders a neutral 50/50 bar). */
  cp: number | null;
  /** Side to move for the evaluated position, used to convert cp to White POV. */
  sideToMove: Side;
  /** Board orientation, so the bar's White end matches the board's White side. */
  orientation: Side;
}

/** A thin, Lichess-style vertical evaluation bar, themed to the Tournament Hall
 *  palette. White's advantage fills cream from White's side; Black's fills walnut
 *  from the top. Sits to the LEFT of the board and flips with board orientation. */
export default function EvalBar({ cp, sideToMove, orientation }: EvalBarProps) {
  const cpWhite = cp == null ? null : sideToMove === "white" ? cp : -cp;
  const whitePct = cpWhite == null ? 50 : whiteWinPct(cpWhite);
  const whiteLeads = cpWhite == null ? true : cpWhite >= 0;

  // White's physical end of the bar: bottom normally, top when the board is flipped.
  const whiteEnd: "top" | "bottom" = orientation === "white" ? "bottom" : "top";
  const leaderEnd: "top" | "bottom" = whiteLeads
    ? whiteEnd
    : whiteEnd === "bottom"
      ? "top"
      : "bottom";

  // Fill the whole bar and scale it from White's end: a compositor-only transform
  // (no height/layout animation), so the fill glides when the eval changes.
  const fillStyle: CSSProperties = {
    backgroundColor: "var(--board-light)",
    transform: `scaleY(${whitePct / 100})`,
    transformOrigin: whiteEnd,
    willChange: "transform",
  };

  const labelStyle: CSSProperties = {
    color: whiteLeads ? "var(--signal-ink)" : "var(--board-light)",
  };
  labelStyle[leaderEnd] = 3;

  const evalTitle =
    cpWhite == null
      ? "Evaluation unavailable"
      : `Engine evaluation: ${magnitudeLabel(cpWhite)} for ${whiteLeads ? "White" : "Black"}`;

  return (
    <div
      className="relative w-[16px] shrink-0 self-stretch overflow-hidden rounded-[6px] border sm:w-[18px]"
      style={{ backgroundColor: "var(--board-dark)", borderColor: "var(--border)" }}
      role="img"
      aria-label={evalTitle}
      title={evalTitle}
    >
      {/* White's advantage: cream fill anchored to White's end of the bar. */}
      <div
        className="absolute inset-0 transition-transform duration-300 ease-out motion-reduce:transition-none"
        style={fillStyle}
      />
      {/* Brass hairline at the 50% (equal) mark. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0"
        style={{
          top: "50%",
          height: "1px",
          transform: "translateY(-0.5px)",
          backgroundColor: "color-mix(in oklab, var(--signal) 70%, transparent)",
        }}
      />
      {/* Numeric eval at the leader's end (dark ink on cream, cream on walnut). */}
      {cpWhite != null && (
        <span
          aria-hidden
          className="tnum absolute inset-x-0 text-center text-[9px] font-semibold leading-none sm:text-[10px]"
          style={labelStyle}
        >
          {magnitudeLabel(cpWhite)}
        </span>
      )}
    </div>
  );
}
