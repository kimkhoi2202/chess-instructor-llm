# BrainLift Handoff Prompt — Chess Coach Behavior Thesis

> Paste everything below the divider into a fresh agent in this same repo
> (`/Users/khoilam/Documents/Crossover/coding-projects`). It is fully
> self-contained: the fresh agent has no prior conversation context.

---

You are picking up an **in-progress BrainLift** (a spiky-POV research artifact) for a fine-tuned chess-coaching project. You have **no prior conversation context**, so this prompt gives you everything: the product, the thesis, where every file lives, what is done vs open, the methodology you can invoke, and a menu of tasks. **Do not invent facts** — everything you assert must trace to a file listed here or to a primary source. Ground every number in the evidence files. Be honest: this project does **not** claim the small model beats the frontier at coaching, and you must not drift into that overclaim.

Before doing anything substantive, **read** `brainlifts/chess-coach-behavior-thesis/brainlift.md` (the shipping doc) and the evidence files in section 5, then confirm with me which task from the menu (section 7) you should run. I will pick and steer.

## 0) What a BrainLift is (30-second version)

A BrainLift is a research artifact that produces a **depth-gated menu of candidate-valid Spiky POVs (SPOVs)** — off-consensus, defensible, falsifiable stances — NOT one proven winner. It is organized by Depth of Knowledge: DOK 1 facts → DOK 2 knowledge tree → DOK 3 insights → DOK 4 SPOVs. Each SPOV carries a crux (the one falsifiable claim that would collapse it), a disconfirmer, and a strength tier (Validated / Strong / Weak). The human is the one who runs the tests; the pipeline's job is to hand them the largest honest set of testable stances. Full methodology is in the skills named in section 6.

## 1) Project context — the product and the money/product goal

**Product.** `chess-instructor-llm` is a **level-calibrated, engine-grounded chess coach** built on a **fine-tuned Qwen3-1.7B** (4-bit MLX) open model. The user sets a position, marks a move they were unsure about, and picks a rating tier (Beginner 1000–1200 / Intermediate 1300–1600 / Advanced 1700–2000). The coach returns **one sound teaching move** with a plain-language explanation — **no centipawns, no engine jargon, no GM-only lines**. Every recommendation is grounded in **Stockfish** (a pool of *sound* moves + short lines) and **Maia** (how likely a human at that rating is to find each move), and each board claim is checked by a **non-LLM faithfulness verifier** against the real position before it reaches the student. There is a running local platform ("The Analysis Room": FastAPI backend + Next.js front end, `./run_platform.sh`).

**The bet (NOT "a 1.7B plays better chess than a frontier model" — it never will).** The narrow, measurable claim is: **one specific behavior — leveled, human-findable "teaching-move" coaching in a steady no-engine-speak voice — is not reliably delivered by a prompted frontier model, and can be trained into a small model to run reliably, cheaply, and locally.**

**Money / product goal (the honest win).** The small local model's defensible advantage is **form factor + register consistency**: ~$0 marginal inference cost (local 4-bit MLX), **private** (data stays on-device), **offline**, low-latency, and a low-variance jargon-free voice. Concretely: running the 100-position 5-model benchmark cost ~$24 total, of which the two local models (ours + base) were **$0.00** while the three frontier models cost the rest per API pricing. The economics are treated as a **secondary, low-confidence** advantage — real in kind, never used as a load-bearing number.

**Published artifacts (v1 baseline):** model `khoilamalphaai/qwen3-1.7b-chess-coach-mlx`, dataset `khoilamalphaai/chess-coach-benchmark`, and a results Space, all on Hugging Face Hub under `khoilamalphaai`.

## 2) The BrainLift's core thesis (as it actually appears in `brainlift.md`)

**Title of the shipping doc:** "What a Small Fine-Tuned Model Actually Buys a Chess Coach: Grounding Carries Dependability, the 1.7B Carries Form Factor." **Owner:** Khoi Lam. Dated July 6, 2026.

**Original question (verbatim intent):** *Can a small (~1.7B) open model, fine-tuned on engine-grounded distilled data, deliver reliably level-calibrated chess coaching — the most instructive move for a student's rating, explained without leaking engine internals — more dependably than a well-prompted frontier model?*

**How it evolved (the reframe, after four isolated critics converged).** Dependability in this kind of system is **not carried by the model weights**. It is carried by three parts that sit **outside** the language model: (1) a strong **engine (Stockfish)** that certifies which move is sound, (2) **tactical/positional detectors + grounding** that expose the concrete features of the position as verified facts, and (3) a **non-LLM verifier** that checks each explanation claim against the engine and the board before anything reaches the student. The **fine-tuned 1.7B model is the "last-mile compressor"**: it renders that already-grounded, already-verified behavior locally, cheaply, privately, and in a steady low-variance no-engine-speak voice. It is **not the origin of dependability.** The honest answer to the core question is a **conditioned yes**: with grounding held constant, a small fine-tuned model can *match* a well-prompted frontier model on faithful level-fit and *win* on form factor and register consistency — but it is **not** more dependable than a frontier model in general, and the sharper "wins on the worst-case tail" claim is a real bet only the project's own grounding-held-constant comparison can settle.

**Two measurement rules that fall out of the thesis:** (a) "more dependable" must be measured as the **worst-case rate at which every constraint passes at once** under repeated sampling at the deployment temperature — not an average quality score; (b) **faithfulness must be gated by the non-LLM checks first**, and any remaining tone/level scoring must come from a **different model family** than the one that produced the text (a model blesses writing shaped like its own).

**In scope / out of scope** are stated in `brainlift.md`'s Purpose; keep to them.

### The SPOV menu that survived (a menu, not a winner)

The doc ships **12 SPOVs**, tiered by how strong the backing is *right now*:

- **Validated (established by outside primary evidence AND still off-consensus):**
  - **SPOV 1** — An unaided explanation judge, and especially a same-family one, systematically passes false chess coaching as truthful, so **faithfulness must be gated by a non-LLM engine-and-detector check before any preference/pedagogy score counts.**
  - **SPOV 2** — **Grounding the move does not ground the explanation.** Even with the move engine-verified as best, a small model narrating the reasons still invents tactics; what removes fabrication is a **claim-level non-LLM check**, not a verified move.
- **Strong (candidate-valid; decisive resolution waits on the project's own grounding-held-constant run):**
  - **SPOV 3** — When a small local coach appears to beat a frontier model, what it actually wins is **form factor, not dependability**; give a prompted frontier the same grounding/schema/rubric and it matches or beats the small tuned model on truthful level-fit.
  - **SPOV 4** — At fixed grounding, the small tuned model's honest capability win is **register consistency, not truth** (lowest-variance renderer of plain no-engine-speak voice).
  - **SPOV 5** — The shippable dependability asset is the **detector-and-verifier layer, not the dataset or the weights**; the fine-tune is the finisher, and its value must survive an ablation.
  - **SPOV 6** — This "translate verified truth into leveled language" recipe is safe to ship **only where the reasoning, not just the answer, has a cheap non-LLM verifier**; chess is deceptive because answer-verifiability masquerades as rationale-verifiability. *(Needs an external multi-domain study.)*
  - **SPOV 7** — One output, **two independent axes**: register is weight-learnable (the fine-tune can own it); faithfulness is not reliably weight-learnable without an external verifier. A single blended "coach quality" score is a category error.
  - **SPOV 8** — Fine-tuning on **raw, unfiltered** frontier transcripts risks being worse than base-plus-prompt (it imitates confident-assertion style and can amplify fabrications); the fix is **faithfulness-filtered data / faithfulness-as-reward**, not more transcripts.
  - **SPOV 9** — **Leveling is largely handled** by grounding + a simple tier rubric, so **faithfulness, not level-calibration, is the hard open axis**; effort spent polishing human-likeness while fabrication goes unfixed is allocated backward.
  - **SPOV 10** — Given an incomplete verifier, the safer coach is **coverage-bounded**: assert only detector-verifiable claims and abstain otherwise. "Say less, truthfully" beats "say more, fluently."
  - **SPOV 11 (flagged riskiest)** — **"More dependable" = worst-case, all-constraints-at-once pass rate under repeated deployment sampling, not mean quality.** The small tuned model's plausible edge lives on that tail; the affirmative half has **no supporting evidence yet** (what's solid is the measurement definition).
- **Weak / supporting:**
  - **SPOV 12** — **Human-move modeling (Maia) is a descriptive learner signal, not the pedagogical objective**: it says what a rated player would *play*, not what should be *taught*; using human-likelihood as the sole selector can drill misconceptions. *(Softened from a stronger form that was refuted — see rejected.)*

> Note on numbering: `brainlift.md` orders the menu **Validated first** (its SPOV 1/2 = the faithfulness-gate and move-≠-explanation claims). The working file `03-spov-candidates.md` uses a different numbering (its SPOV 1 = "the system, not the fine-tune, is the deliverable"). When you cite an SPOV, quote its text, not just its number.

### What was rejected (and why) — do not silently resurrect these

- **SPOV 5 strong form** — "Maia tells you NOTHING about what to teach and using human-likelihood as the selector ACTIVELY HARMS learners." REJECTED (Test 2 leaky + Test 3 crux refuted): primary evidence shows human-likelihood models are **useful-but-not-sufficient**, not harmful. Retained only as the softened Weak SPOV 12.
- **DOK 3 "C1-vs-Play-Magnus resolves as benchmark-vs-deployment"** — REJECTED: it's a truism, and its "same-lab contradiction" premise is factually false (C1 is CSSLab; Play Magnus / Take Take Take is a separate product team that merely uses CSSLab's Maia).
- **Protocol decoy** — "A chess coach should adapt to the student's level" was run as a deliberate truism and correctly caught (both cold testers rated it mainstream), confirming the gate tracks spikiness, not rhetoric.

## 3) Where everything lives (exact absolute paths)

**BrainLift run folder:** `/Users/khoilam/Documents/Crossover/coding-projects/brainlifts/chess-coach-behavior-thesis/`

File inventory (role — state):

- `brainlift.md` — the **final shipping doc** (Nessie template: Title → Owner → Purpose → DOK 4 SPOVs → Experts → DOK 3 Insights → DOK 2 Knowledge Tree). **Done** (assembled, self-contained).
- `00-intake.md` — the core question, project context, and first-principles map. **Done.**
- `01-experts-and-facts.md` — pooled DOK 1 facts + candidate experts, tagged by researcher lane. **Done.**
- `01b-citations.md` — the citation gate: per-cluster verification, FIXes, and the load-bearing GAPS (the ~1.7B affirmative is under-evidenced). **Done** (zero drops, a few fixes, gaps flagged as the crux-to-test).
- `01c-expert-signals.md` — named experts' recent public positions as verified, opinion-tagged DOK 1 facts, + the sharpest expert tensions. **Done.**
- `02-summary-insights.md` — DOK 2 Knowledge Tree + first-pass DOK 3 insights. **Done** (superseded by 02b for the validated insight set).
- `02b-insights-validated.md` — the **post-critique** insight set after four isolated critics converged on the "grounding carries dependability" reframe, + residual gaps. **Done.**
- `03-spov-candidates.md` — the wide pool of 12 candidate SPOVs, each with assertion / core question / disconfirmer / reach, anchored to the base eval. **Done.**
- `04-spov-validation.md` — the 3-test SPOV Testing Protocol results (cold spikiness, defensibility crux+depth, primary-source verification) + the final tiered menu + the decoy check. **Done.**
- `rejected.md` — the reject log (what was cut at each DOK gate and why). **Done** (see section 2 above).
- `child-brainlifts.md` — the 7 big/contested sub-topics detected during research, recorded as **future** child BrainLifts (resolved in-run this pass, not spawned). **Open lever** — any of these can be spun off as its own full BrainLift.

> There is **no `research/` subfolder** in this run (some sibling BrainLift runs have one; this one does not).

**Empirical evidence the BrainLift rests on** (all under `/Users/khoilam/Documents/Crossover/coding-projects/chess-instructor-llm/`):

- `RESULTS.md` — base-vs-tuned, cross-family Claude judge, held-out scenarios (the "truthfulness is flat" result).
- `RESULTS_BENCHMARK.md` — the 5-model × 2-condition (grounded/ungrounded) × 100-position benchmark, blinded cross-family council (labels the tuned model `chess-coach-v1`; generated 2026-07-06 23:15 UTC).
- `RESULTS_BENCHMARK_v2.md` — the **newer** run of the same benchmark on the retrained `chess-coach-v2` model (generated 2026-07-07 00:45 UTC). Also `RESULTS_BENCHMARK_v1.md` exists.
- `data/analysis/GAP_REPORT.md` — proves the frontier is weak at the narrow leveled-teaching-move behavior (and strong on truthfulness) on 50 held-out positions.
- `data/analysis/DIVERGENCE_REPORT.md` — the tuned model's tier-differentiated move-SELECTION analysis (weak + mis-directed; 0% contrastive tier pairs).
- `data/analysis/PUZZLES_REPORT.md` — Lichess puzzle solutions are engine-best, not findable labels (a data-sourcing decision).
- `SUBMISSION.md`, `README.md` — deliverables map, product architecture, win-condition scorecard, honest gaps, v2 roadmap.
- `docs/EVAL_AND_ITERATE.md` — the eval protocol and the v2 pass bar.
- `prompts/` (`coach_system.md`, `teacher_system*.md`, `principles.md`, `tier_guides.md`, `eval_rubric.md`), `config/schema.py`, `src/engine/faithfulness.py` — the actual grounding + verifier + rendering the thesis describes.

## 4) Current state — complete vs open

**Complete:** The full BrainLift pipeline ran end to end. `brainlift.md` is assembled in the Nessie template with 12 tiered SPOVs (2 Validated, 9 Strong, 1 Weak), 11 experts, 6 DOK 3 insights, and a DOK 2 knowledge tree over ~120 reviewed sources. Citation gate passed (zero hallucinated/fabricated citations, zero drops). All working artifacts exist and are internally consistent.

**Open (the honest gaps, straight from the artifacts):**

- The **~1.7B affirmative is a candidate bet, not a settled fact** — the strongest grounded small-model wins in the literature are at 4B (C1) / 7–8B (MATE); there is a ≤3B "learnability gap" headwind. The project's own base-vs-tuned run is the intended disconfirmer.
- **SPOV 11's affirmative half is unmeasured** — worst-case pass-all-constraints under **repeated sampling at deployment temperature** has not been run (evals to date are greedy / small-n). The metric definition is solid; the "small model wins the tail" claim is a live bet.
- **SPOVs 3, 4, 12** (grounding-held-constant head-to-head; register variance under k-sampling) resolve on the project's own comparison, which is now **partially available** (see the new benchmark evidence in section 5) but not yet in a k-sampled, variance-first form.
- **SPOV 6 & the softened SPOV 12** need **external studies** (multi-domain rationale-verifiability; a Maia-selector learning study).
- The **7 child-BrainLift sub-topics** are recorded as future runs, not resolved.
- **The `brainlift.md` numbers were written against the EARLY base run** (n=9 greedy, gpt-5.5-pro judge: truthfulness 0.00, no-engine-speak 0.11, move-sound 1.00, spec-adherence 0.875, level-calibration 0.875). **Richer, newer evidence now exists** (section 5) that bears directly on several SPOVs and is **not yet folded into `brainlift.md`.** This is the single biggest "refresh" opportunity.

## 5) The empirical evidence (cite these exact numbers; do not round loosely)

**A) Early base run (what `brainlift.md` currently cites).** Untuned Qwen3-1.7B-4bit, 9 greedy samples, gpt-5.5-pro judge: **move-sound 1.00, spec-adherence 0.875, level-calibration 0.875, no-engine-speak 0.11, truthfulness 0.00, task-quality 0.00.** The failures are concentrated in **fabricated tactics and leaked engine talk, not move selection** — the load-bearing receipt for "dependability is a system property; register and faithfulness are separate axes."

**B) Base → tuned (cross-family Claude judge, held-out; `RESULTS.md`).** Objective: **move_sound 87%→100%, no_engine_speak 33%→100%, ply_cap_ok 67%→100%.** Judge (0–2): **spec_adherence 0.47→0.93, level_calibration 0.60→1.13, no_engine_speak 0.87→1.87, truthfulness 0.13→0.13 (FLAT), task_quality 0.13→0.27.** Reading: the fine-tune wins decisively on everything it can control by shaping the training distribution (style/format/register), and **truthfulness is the one flat axis** — the tuned model says the right kind of thing in the right voice but invents board facts, because the v1 labels were filtered for format/soundness but **not faithfulness**.

**C) 5-model × 2-condition × 100-position benchmark (`RESULTS_BENCHMARK.md`, v1 model).**
- **Grounding lifts move-soundness:** OURS 36%→100%; BASE 28%→92%; frontier avg 50%→100%.
- **Grounded fabrication rate:** OURS **93%→38%**; frontier avg 5%→3%. (Grounding is the lever a 1.7B can't supply from its own board-tracking, but it does **not** close the gap to the frontier.)
- **No-engine-speak:** OURS 100%/100% (the fine-tune owns the style gate).
- **Council instructiveness (mean rank, 1=best of 5):** OURS 4.44→**4.19**; grounded frontier avg **~2.10** (best grounded = Claude 1.93). **Grounding narrows the coaching gap but does not erase it — a bigger model still explains more instructively.** This is the real, honest, open gap.
- **Judge self-preference:** mean signed **+0.44 rank** (small vs the gaps → council ranking isn't just lab loyalty). *(This is the project's own empirical measurement of the same-family-judge risk behind SPOV 1.)*

**C2) Newer benchmark on the retrained v2 model (`RESULTS_BENCHMARK_v2.md`).** Same design, `chess-coach-v2`: move_sound OURS 41%→98%; **fabrication OURS 99%→33%** (frontier 5%→3%); council OURS 4.11→**3.68** vs grounded frontier avg **~2.21**; self-preference +0.43. (Directionally identical story; v2 shifts the ours numbers slightly.)

**D) Frontier gap (`GAP_REPORT.md`, 50 held-out, grounding byte-identical to the app).** Frontier averages: **tier-differentiation 22.7%** (picks a different move across the three tiers), **engine-mirroring at every tier 68.7%** (returns Stockfish's #1 at all tiers, blind to level), **beginner-findability on the opportunity subset 20.5%**. **Honest counter-finding:** on truthfulness the frontier is *strong* — it fabricates a board fact in only **3.3%** of outputs vs the **v1 tuned model's 51.3%**. So the frontier is weak at leveling the move but strong at not lying about the board; any credible small-model win must clear **both** bars.

**E) Move-selection divergence (`DIVERGENCE_REPORT.md`, 120 held-out, tuned v1, greedy).** **Tier-differentiation 25.0%** (30/120), and **mis-directed** (beginners get the sharp engine move slightly *more* than advanced). Mechanism: model pick == Stockfish best **76.7%** (it regresses toward pool[0] *more* than the teacher's 65.8%), and the training data has **0% contrastive tier examples** (0 of 2,132 FENs taught at more than one tier). The model's distinctive, reliable value is the **explanation**, not tier-differentiated move selection.

**F) Puzzle labels (`PUZZLES_REPORT.md`, 150 puzzles).** Lichess puzzle solutions equal Stockfish's #1 move **99.3%** of the time and sit in the sound pool **100%** (median cp-loss 0), but at a beginner (Maia-1100) only **60%** are naturally findable and **12.7%** are hard-to-find. Verdict: puzzles are **fuel** (positions + motif coverage), **not** drop-in coaching labels — external solutions are context, never labels.

**Why this matters for your task:** several SPOVs get *stronger* under the new evidence — e.g., SPOV 3 (the grounded head-to-head now directly shows the frontier still out-instructs the small model), SPOV 2/5 (grounding drops but does not zero fabrication; move-sound 100% with fabrication 33–38%), and SPOV 1 (the project now has its own +0.44 self-preference measurement). Others get *complicated* — e.g., SPOV 9 ("leveling largely handled") is true for explanation register but **not** for tier-differentiated move *selection*, which `DIVERGENCE_REPORT.md`/`GAP_REPORT.md` show is weak and mis-directed. Fold this in carefully and honestly.

## 6) Methodology + skills you can invoke

The BrainLift pipeline and all definitions live in these two skills (read them before editing the doc so your changes stay faithful to the format and gates):

- `/Users/khoilam/.cursor/skills/brainlift/SKILL.md` — the **autonomous** multi-agent pipeline: 6 cross-family researchers → citation gate → expert-signal scan → child-BrainLift gate → synthesize (DOK 2/3) → DOK 3 debate → generate DOK 4 SPOV candidates → the **3-test SPOV Testing Protocol** (Test 1 cold spikiness across families, Test 2 adversarial crux-hunt + depth gate, Test 3 primary-source verification) → assemble in the Nessie template. Key rule: **keep every candidate-valid SPOV as a tiered menu; never crown one winner; never lower the gate.** Tiers (Validated / Strong / Weak) come only from the gate + external primary-source check.
- `/Users/khoilam/.cursor/skills/brainlift-guided/SKILL.md` — the **interactive** variant (same pipeline, same gates) that adds an intake interview and mid-run checkpoints so the menu fits the user's situation. **This is the one to use if the user wants to be asked questions / steered while you work.**

**Nessie / House style for `brainlift.md`** (from the base skill): self-contained submission artifact — **no references to working files** (`01-experts-and-facts.md`, `rejected.md`, etc.) or agent/step names inside it; **no em dashes**; `Owner: [name]`; plain high-school-level prose; Title → Owner → Purpose (+ In/Out of Scope) → DOK 4 SPOVs → Experts → DOK 3 Insights → DOK 2 Knowledge Tree. Preserve this if you edit the shipping doc.

## 7) Task menu — pick with me before you run (I will steer)

Propose which of these you'll do and confirm scope with me first. Likely options:

1. **Finalize / tighten the shipping doc** — copy-edit `brainlift.md` for concision, House-style compliance (no em dashes, no working-file references), and internal consistency, without changing any tier or claim. Lowest-risk polish.
2. **Deepen a specific SPOV** — pick one (e.g., SPOV 2, 3, 5, 6, or the riskiest 11) and strengthen its elaboration, crux, disconfirmer, and reach with tighter evidence; optionally re-run the relevant part of the Testing Protocol.
3. **Add the latest benchmark evidence** *(highest-leverage refresh)* — fold `RESULTS.md`, `RESULTS_BENCHMARK.md` (+ `_v2`), `GAP_REPORT.md`, `DIVERGENCE_REPORT.md`, `PUZZLES_REPORT.md` into the DOK 2 knowledge tree and the affected SPOV "how to resolve it" notes. Update the doc's base-run receipts to reflect the newer cross-family and grounded head-to-head numbers, and flag where the new evidence **strengthens** vs **complicates** an SPOV (see section 5). Keep the tiers honest — don't promote to Validated without the external primary-source check the protocol requires.
4. **Adapt to the Nessie / BrainLift template** — if the target is a specific Nessie output format, reshape `brainlift.md` to match while preserving every candidate-valid SPOV and its tier.
5. **Sanity-check spikiness** — re-run Test 1 (cold, cross-family, SPOV-sentence-only) on any SPOVs you touched to confirm they're still off-consensus (not truisms) and re-confirm cruxes against primary sources.
6. **Spin off a child BrainLift** — take one of the 7 recorded sub-topics in `child-brainlifts.md` (e.g., "fine-tune vs prompt for reliable level-calibration," "does engine-grounding neutralize small-model hallucination transfer," "LLM-as-judge validity for this comparison") and run it as its own full BrainLift.

## 8) Guardrails (do not violate)

- **Ground every claim** in the files listed here or a primary source; cite the source. If a fact isn't in the evidence files or a retrievable primary source, don't assert it.
- **Be honest — do not overclaim.** The small model does **not** beat the frontier at coaching. The blinded council still ranks the frontier well above the 1.7B on instructiveness even **with** identical grounding (ours ~4.19 grnd vs frontier ~2.10), and the frontier fabricates far less (3.3% vs the tuned model's 51.3% raw). The small model's real, defensible wins are **form factor + register consistency**. The "more dependable on the worst-case tail" claim (SPOV 11) is an **open bet with no affirmative evidence yet** — keep it labeled as such.
- **Keep the menu a menu.** Never collapse the SPOVs to a single winner, never delete a candidate-valid SPOV by preference, and never inflate a tier — Validated / Strong / Weak come only from the gate + the external primary-source check (an agreeing model is not evidence; documented sycophancy is 46–95%).
- **Don't resurrect rejected claims** without new primary evidence (see section 2 — the Maia "actively harms" strong form, the false "same-lab contradiction," the decoy).
- **Preserve House style** in `brainlift.md` (no em dashes, no working-file/agent references, plain prose, Nessie order).
- **Separate the two truthfulness numbers** when you cite them: the model's **raw** fabrication rate (51.3% v1 ungrounded; 33–38% grounded) vs the **production** defense (the live verify-and-regenerate gate in `src/engine/faithfulness.py`). Don't conflate them.
- **Economics stays secondary / low-confidence** — real in kind, never a load-bearing number.

**First action:** read `brainlifts/chess-coach-behavior-thesis/brainlift.md` + the section-5 evidence files, then tell me which task (section 7) you propose and your plan. Wait for my pick before making edits.
