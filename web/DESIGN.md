# Design

Visual system for the AI Chess Instructor. Product register: design SERVES the
task. This document is the source of truth for the revamp — it deliberately
departs from the previous "warm analysis room" (warm-green ink, brass lamp,
parchment, serif prose, IBM Plex Mono), which read as the escape-slop
editorial/warm-study reflex plus the chess-serif cliché.

## Theme

**The physical scene (one sentence):** a focused club player at a dark desk late
at night, replaying the game they just lost on one calm glowing screen — a bench
instrument that *measures* the position and hands back a single, confident
verdict; the mood is quiet concentration, not play.

That scene forces **dark** (night, single screen, concentration) and forces
**cool** (an instrument bench, not a cozy lamp). The aesthetic lane is a
**precision measuring instrument / analysis console** — quiet cool graphite,
hairline structure, tabular readouts — decisively *not* wooden-and-serif and
*not* cream-editorial.

## Color

**Strategy: Restrained** (product floor) with one committed signal accent whose
meaning is load-bearing. The palette encodes the product thesis: **cool = measured
engine truth; one warm signal = the human verdict / the move you should focus on.**
All values OKLCH; tinted neutrals lean cool (toward the ink's own hue); never pure
`#000` / `#fff`.

### Core ramp (dark, the only theme)
| Token | OKLCH | Role |
|---|---|---|
| `--bg` | `oklch(0.171 0.012 258)` | app canvas — cool graphite, faint blue tint |
| `--surface` | `oklch(0.216 0.014 258)` | panels, cards |
| `--surface-2` | `oklch(0.252 0.016 258)` | raised rows, inputs |
| `--surface-3` | `oklch(0.288 0.017 258)` | hover / active fills |
| `--ink` | `oklch(0.948 0.006 258)` | primary text (cool near-white) |
| `--muted` | `oklch(0.735 0.014 258)` | secondary text — verified ≥ 4.5:1 on `--bg`/`--surface` |
| `--faint` | `oklch(0.62 0.014 258)` | tertiary / captions (large or non-essential only) |
| `--border` | `oklch(0.34 0.012 258)` | hairline structure |
| `--separator` | `oklch(0.30 0.010 258)` | quiet rules |

### Signal + semantics
| Token | OKLCH | Role |
|---|---|---|
| `--signal` | `oklch(0.80 0.163 66)` | **the accent** — luminous instrument amber. The recommended move, current selection, focus ring, primary action. Reserved; never decoration. |
| `--signal-ink` | `oklch(0.22 0.03 66)` | text/icon on a `--signal` fill |
| `--engine` | `oklch(0.72 0.09 236)` | cool "engine data" tint — raw Stockfish/Maia readouts, the neutral measured lane |
| `--your-move` | `oklch(0.66 0.15 25)` | the move the *user* played (distinct from the recommendation) |
| `--caution` | `oklch(0.80 0.12 78)` | inaccuracy / warning |
| `--good` | `oklch(0.74 0.12 158)` | sound / positive (used sparingly; NOT the brand color — reach past chess-green) |

Contrast is verified against the surface each token sits on. Meaning is never
color-only: the recommendation carries the word "recommends" + an arrow, "your
move" carries a label, engine data is a labeled section.

## Typography

Full replacement of the previous stack. Two families, paired on a real structural
contrast axis (proportional grotesque vs. monospace), mono strictly **earned**
(chess notation, eval, FEN — real tabular data, not developer costume).

- **Archivo** (variable grotesque) — UI, labels, headings, coaching prose, wordmark.
  A technical, mechanical grotesque that reads like precise instrument signage.
  Replaces both Hedvig serif (kills the chess-serif cliché — coaching prose is now
  grotesque, not literary) and Hanken Grotesk. Hierarchy comes from **weight +
  width + tracking**, not a serif/sans split.
- **Spline Sans Mono** (variable monospace) — the notation face: the big recommended
  move (notation-as-hero), eval numbers, engine principal variations, FEN input.
  Replaces IBM Plex Mono (a reflex-reject default). Tabular figures always on for
  data that must not reflow.

### Scale (fixed rem, modular ≈ 1.25)
| Step | Size | Use |
|---|---|---|
| `--text-verdict` | clamp(2.6rem, 6vw, 3.4rem) | the recommended move (mono, hero) |
| `--text-2xl` | 1.75rem | idle headline |
| `--text-xl` | 1.375rem | panel titles / prose lead |
| `--text-lg` | 1.125rem | coaching prose body |
| base | 1rem (16px) | UI body |
| `--text-sm` | 0.875rem | secondary UI |
| `--text-xs` | 0.75rem | smallest labels (floor — no 9–11px body text) |

- Body / prose measure capped at ~66ch.
- Display tracking floor ≥ -0.03em; labels use modest positive tracking (≤ 0.06em),
  **not** the wide 0.2em all-caps eyebrow (that tell is removed).
- Section labels are quiet sentence/short-form, used with restraint — not a
  repeated tiny-uppercase-tracked kicker over every block.

## Layout

- Two-column console on desktop: **board is the hero** (left), coaching + analysis
  console (right). Single stacked column on mobile, board first.
- Max width ~1240px; generous outer padding; varied vertical rhythm (tight groups
  inside panels, generous separation between them).
- Hairline structure over boxes: prefer thin `--border`/`--separator` rules and
  quiet surface steps to heavy nested cards. No nested cards. Card radius 12–14px
  (no 24px+ over-rounding). Full borders or a single defined shadow — never a 1px
  border *and* a big soft shadow on the same element.
- Board frame: a quiet cool bezel (surface + hairline + one defined shadow); the
  chessground board itself keeps its Lichess look (brown/cburnett).

## Components

Every interactive control ships default / hover / focus-visible / active /
disabled. States:
- **Loading:** calm skeleton/measured shimmer in the console; board dims with a
  quiet status line. No spinner-in-the-middle-of-content as the only signal.
- **Idle:** teaches the instrument (what to do), reads as ready, not empty.
- **Error:** plain-language, names the problem and the recovery, preserves context.
- **Recommended move:** the single loudest element — `--signal`, large mono
  notation, arrow. Everything else is subordinate.

## Motion

- 150–250ms, ease-out (quart/expo). One purposeful staggered coaching reveal when
  a verdict lands (it conveys "the answer is arriving"), transform/opacity only.
- Every animation has a `prefers-reduced-motion` path to an instant, already-visible
  state. No decorative or layout-property motion.

## Bans (enforced)

Removed / forbidden: `feTurbulence` paper-grain, gradient text, decorative glass,
side-stripe borders, hero-metric template, identical icon-card grids, repeated
tiny-uppercase-tracked eyebrows, serif-display-as-chess-voice, IBM Plex/Space Mono,
warm cream/parchment canvas, `border-1px + big-shadow` ghost cards, 24px+ card
radius, meaning-by-color-alone.
