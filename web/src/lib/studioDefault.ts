// AUTO-GENERATED cached-first seed for the Studio homepage.
//
// The DEFAULT king-and-pawn endgame (id vaLVwTHK_77) with the tuned coach's
// PRECOMPUTED answer at all three rating tiers, so the Studio renders the
// tier-adaptive move (Ne6+ / h6 / Kd2) INSTANTLY on mount without any live
// call to the scale-to-zero Modal endpoint (which cold-starts in ~2-3 min).
//
// Provenance (all values are precomputed, never fabricated at runtime):
//   - per-tier recommended move + coaching prose + takeaway: web/public/showcase.json
//     (OURS-v4 cell for this position; the same tuned model the live demo serves)
//   - position-level engine facts (sound pool, best move, the student's Ne2 +
//     severity) and Maia human-frequency: web/public/library.json (this exact FEN)
//     -- these depend only on the position, so they are shared across the tiers,
//     exactly as the live /api/coach_all computes them once per position.
//
// Regenerate by re-merging those two public JSON files for this position.
import type { CoachResponse, Tier } from "@/lib/api";

export const STUDIO_DEFAULT_FEN = "8/7b/5p2/P1kp3P/2pN1P2/4K3/8/8 w - - 1 39";
export const STUDIO_DEFAULT_STUDENT_UCI = "d4e2";

/** Precomputed tuned-coach answers for the default position, one per tier. */
export const STUDIO_DEFAULT_TIERS: Record<Tier, CoachResponse> = {
  "beginner": {
    "recommended_move_san": "Ne6+",
    "recommended_move_uci": "d4e6",
    "coaching": "I'd play Ne6+. It brings the knight to e6 while covering the pawn on f4 and gives check.",
    "takeaway": "A check with a point forces your opponent to react on your terms.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "Kd2",
      "best_cp": 258,
      "sound_pool": [
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 258,
          "pv": [
            "Kd2",
            "Kxd4",
            "a6",
            "c3+",
            "Kd1",
            "Kd3"
          ]
        },
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 252,
          "pv": [
            "h6",
            "Bg8",
            "Kd2",
            "Bh7",
            "Ne6+",
            "Kb5"
          ]
        },
        {
          "san": "Ke2",
          "uci": "e3e2",
          "cp": 249,
          "pv": [
            "Ke2",
            "Bd3+",
            "Ke3",
            "Bh7",
            "h6",
            "Bg8"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 243,
          "pv": [
            "f5",
            "Bg8",
            "h6",
            "Bh7",
            "Ke2",
            "Kd6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 240,
          "pv": [
            "Ne2",
            "Kb5",
            "Nc3+",
            "Kc5",
            "Na4+",
            "Kb5"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 157,
          "pv": [
            "Ne6+",
            "Kb5"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 18,
        "severity": "none"
      }
    },
    "maia": [
      {
        "san": "a6",
        "uci": "a5a6",
        "policy": 0.3784
      },
      {
        "san": "Ne6+",
        "uci": "d4e6",
        "policy": 0.2464
      },
      {
        "san": "h6",
        "uci": "h5h6",
        "policy": 0.1204
      },
      {
        "san": "f5",
        "uci": "f4f5",
        "policy": 0.1074
      },
      {
        "san": "Kd2",
        "uci": "e3d2",
        "policy": 0.042699999999999995
      },
      {
        "san": "Ne2",
        "uci": "d4e2",
        "policy": 0.0421
      }
    ],
    "meta": {
      "model": "OURS-v4 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [],
      "attempts": 1,
      "verified_fallback": false
    }
  },
  "intermediate": {
    "recommended_move_san": "h6",
    "recommended_move_uci": "h5h6",
    "coaching": "I'd play h6. Your Ne2 was understandable because it defends the pawn from f4, but it spends a tempo only protecting something you already have. In an endgame like this, active passed pawns matter more than extra defense when nothing is hanging. h6 advances your kingside pawn while keeping your king flexible near its current post on e3. If Black plays ...Bg8, you still have Kd2 available next, so you improve the pawn first without giving up useful king activity. How to find it: Use this routine in practical endgames: first ask, 'Is anything actually loose?' Here, no undefended pieces need emergency help. Next compare protection versus progress: if defending a pawn does not create a new problem, look for whether any pawn can advance safely before making a purely defensive move. Finally, calculate one quiet opponent response and make sure your king still has a useful follow-up. That points you to h6 now, then keep Kd2 ready after ...Bg8.",
    "takeaway": "In safe endgames, do not spend tempi guarding what is fine; activate your passed pawns unless there is an immediate tactic against them.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "Kd2",
      "best_cp": 258,
      "sound_pool": [
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 258,
          "pv": [
            "Kd2",
            "Kxd4",
            "a6",
            "c3+",
            "Kd1",
            "Kd3"
          ]
        },
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 252,
          "pv": [
            "h6",
            "Bg8",
            "Kd2",
            "Bh7",
            "Ne6+",
            "Kb5"
          ]
        },
        {
          "san": "Ke2",
          "uci": "e3e2",
          "cp": 249,
          "pv": [
            "Ke2",
            "Bd3+",
            "Ke3",
            "Bh7",
            "h6",
            "Bg8"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 243,
          "pv": [
            "f5",
            "Bg8",
            "h6",
            "Bh7",
            "Ke2",
            "Kd6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 240,
          "pv": [
            "Ne2",
            "Kb5",
            "Nc3+",
            "Kc5",
            "Na4+",
            "Kb5"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 157,
          "pv": [
            "Ne6+",
            "Kb5"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 18,
        "severity": "none"
      }
    },
    "maia": [
      {
        "san": "a6",
        "uci": "a5a6",
        "policy": 0.3784
      },
      {
        "san": "Ne6+",
        "uci": "d4e6",
        "policy": 0.2464
      },
      {
        "san": "h6",
        "uci": "h5h6",
        "policy": 0.1204
      },
      {
        "san": "f5",
        "uci": "f4f5",
        "policy": 0.1074
      },
      {
        "san": "Kd2",
        "uci": "e3d2",
        "policy": 0.042699999999999995
      },
      {
        "san": "Ne2",
        "uci": "d4e2",
        "policy": 0.0421
      }
    ],
    "meta": {
      "model": "OURS-v4 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [],
      "attempts": 1,
      "verified_fallback": false
    }
  },
  "advanced": {
    "recommended_move_san": "Kd2",
    "recommended_move_uci": "e3d2",
    "coaching": "I'd play Kd2. It brings the king to d2.",
    "takeaway": "Prefer purposeful moves that improve a piece and keep your king safe.",
    "concepts_used": [],
    "side_to_move": "white",
    "engine": {
      "best_san": "Kd2",
      "best_cp": 258,
      "sound_pool": [
        {
          "san": "Kd2",
          "uci": "e3d2",
          "cp": 258,
          "pv": [
            "Kd2",
            "Kxd4",
            "a6",
            "c3+",
            "Kd1",
            "Kd3"
          ]
        },
        {
          "san": "h6",
          "uci": "h5h6",
          "cp": 252,
          "pv": [
            "h6",
            "Bg8",
            "Kd2",
            "Bh7",
            "Ne6+",
            "Kb5"
          ]
        },
        {
          "san": "Ke2",
          "uci": "e3e2",
          "cp": 249,
          "pv": [
            "Ke2",
            "Bd3+",
            "Ke3",
            "Bh7",
            "h6",
            "Bg8"
          ]
        },
        {
          "san": "f5",
          "uci": "f4f5",
          "cp": 243,
          "pv": [
            "f5",
            "Bg8",
            "h6",
            "Bh7",
            "Ke2",
            "Kd6"
          ]
        },
        {
          "san": "Ne2",
          "uci": "d4e2",
          "cp": 240,
          "pv": [
            "Ne2",
            "Kb5",
            "Nc3+",
            "Kc5",
            "Na4+",
            "Kb5"
          ]
        },
        {
          "san": "Ne6+",
          "uci": "d4e6",
          "cp": 157,
          "pv": [
            "Ne6+",
            "Kb5"
          ]
        }
      ],
      "student_move": {
        "san": "Ne2",
        "uci": "d4e2",
        "cp_loss": 18,
        "severity": "none"
      }
    },
    "maia": [
      {
        "san": "a6",
        "uci": "a5a6",
        "policy": 0.3784
      },
      {
        "san": "Ne6+",
        "uci": "d4e6",
        "policy": 0.2464
      },
      {
        "san": "h6",
        "uci": "h5h6",
        "policy": 0.1204
      },
      {
        "san": "f5",
        "uci": "f4f5",
        "policy": 0.1074
      },
      {
        "san": "Kd2",
        "uci": "e3d2",
        "policy": 0.042699999999999995
      },
      {
        "san": "Ne2",
        "uci": "d4e2",
        "policy": 0.0421
      }
    ],
    "meta": {
      "model": "OURS-v4 (Qwen3-32B tuned)",
      "tuned": true,
      "notes": [],
      "attempts": 1,
      "verified_fallback": false
    }
  }
};
