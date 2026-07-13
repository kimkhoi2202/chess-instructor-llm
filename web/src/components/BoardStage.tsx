"use client";

import { useMemo } from "react";
import { Chess } from "chess.js";
import type { DrawShape } from "chessground/draw";
import type * as cg from "chessground/types";
import ChessgroundBoard from "./ChessgroundBoard";
import EvalBar from "./EvalBar";
import type { Orientation } from "@/lib/chess";

export type ArrowKind = "rec" | "student";

export interface StageArrow {
  from: string;
  to: string;
  kind: ArrowKind;
  /** seconds; kept for API compatibility (chessground shapes are instant) */
  delay: number;
  draw: boolean;
}

interface BoardStageProps {
  fen: string;
  orientation: Orientation;
  arrows: StageArrow[];
  lastMove?: string[] | null;
  loading: boolean;
  interactive: boolean;
  onMove: (uci: string) => void;
  /** Best-move centipawns (side-to-move POV) for the coached position, or null
   *  when the board is not on an evaluated position. Drives the left eval bar. */
  evalCp?: number | null;
}

function computeDests(chess: Chess): cg.Dests {
  const dests = new Map<cg.Key, cg.Key[]>();
  for (const m of chess.moves({ verbose: true })) {
    const from = m.from as cg.Key;
    const arr = dests.get(from) ?? [];
    arr.push(m.to as cg.Key);
    dests.set(from, arr);
  }
  return dests;
}

export default function BoardStage({
  fen,
  orientation,
  arrows,
  lastMove,
  loading,
  interactive,
  onMove,
  evalCp = null,
}: BoardStageProps) {
  const { dests, turnColor, check } = useMemo<{
    dests: cg.Dests;
    turnColor: cg.Color;
    check: cg.Color | boolean;
  }>(() => {
    try {
      const c = new Chess(fen);
      const turn: cg.Color = c.turn() === "w" ? "white" : "black";
      return { dests: computeDests(c), turnColor: turn, check: c.inCheck() ? turn : false };
    } catch {
      return { dests: new Map(), turnColor: "white", check: false };
    }
  }, [fen]);

  // Coach arrows → chessground shapes (signal = recommended, yourmove = your move).
  const autoShapes = useMemo<DrawShape[]>(
    () =>
      arrows.map((a) => ({
        orig: a.from as cg.Key,
        dest: a.to as cg.Key,
        brush: a.kind === "rec" ? "signal" : "yourmove",
      })),
    [arrows],
  );

  const cgLastMove = useMemo<cg.Key[] | undefined>(
    () =>
      lastMove && lastMove.length >= 2
        ? [lastMove[0] as cg.Key, lastMove[1] as cg.Key]
        : undefined,
    [lastMove],
  );

  // Accessible description of the board for screen readers (the chessground DOM
  // itself is decorative). States side-to-move, check, and how to interact.
  const boardLabel = `Interactive chess board: ${
    turnColor === "white" ? "White" : "Black"
  } to move${check ? ", in check" : ""}. ${
    interactive
      ? "Drag a piece to play your move; right-click and drag to draw arrows."
      : "Reading the position."
  }`;

  return (
    // Lichess-style eval bar to the LEFT of the board. The bar is a fixed-width,
    // full-height column; the board keeps its square sizing via flex-1 +
    // aspect-square, so the bar never distorts the squares.
    <div className="flex w-full items-stretch gap-2">
      <EvalBar cp={evalCp} sideToMove={turnColor} orientation={orientation} />
      {/* No card/bezel/border around the board: it sits clean on the canvas. */}
      <div className="relative aspect-square min-w-0 flex-1 select-none">
        <ChessgroundBoard
          fen={fen}
          orientation={orientation}
          turnColor={turnColor}
          dests={dests}
          movableColor={interactive ? turnColor : undefined}
          lastMove={cgLastMove}
          check={check}
          autoShapes={autoShapes}
          drawable
          coordinates
          label={boardLabel}
          onMove={(orig, dest) => onMove(`${orig}${dest}`)}
        />

        {loading && (
          <div
            className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 px-6"
            // Light scrim only: the chalk/walnut board stays crisp and legible while
            // the coach reads (a heavy tint here reads as a muddy, broken board). The
            // status label/progress sit in their own contained chip so they're
            // readable over whatever squares they land on.
            style={{ background: "color-mix(in oklab, var(--background) 20%, transparent)" }}
            role="status"
            aria-live="polite"
          >
            <div
              className="flex flex-col items-center gap-2.5 rounded-xl px-5 py-3.5 ring-1 ring-[color:var(--border)] backdrop-blur-sm"
              style={{ background: "color-mix(in oklab, var(--surface) 88%, transparent)" }}
            >
              <span className="text-sm font-medium text-ink">Reading the position</span>
              <span className="relative h-1 w-40 max-w-[52vw] overflow-hidden rounded-full bg-[color:var(--surface-tertiary)]">
                <span className="board-progress absolute inset-y-0 left-0 w-2/5 rounded-full bg-signal" />
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
