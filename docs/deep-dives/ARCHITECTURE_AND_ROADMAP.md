# Architecture Analysis: Issues & LLM-Powered Improvements

**Last updated:** 2026-06-22

---

## Part 0: DreamerV2 Implementation vs. Original Paper

### What Is DreamerV2? (Plain English)

DreamerV2 is a model-based reinforcement learning algorithm that works like this:

1. **The agent builds a mental model of the world** — it learns to predict what will happen next if it takes a certain action, without actually doing it.
2. **It "daydreams" about future scenarios** — using its world model, it imagines many possible futures to figure out which actions lead to good outcomes.
3. **It picks actions based on those daydreams** — a policy network chooses actions and a value network judges how good a situation is.

Think of it like a chess player who mentally simulates moves before touching a piece.

### Current Implementation vs. the Original Paper

The implementation lives in `dreamer/core.py`. The table below reflects the **current live code** as of 2026-04-26. Several parameters that differed from the paper in earlier runs (see TRAINING_NOTES.md v2) have been corrected.

| Component | Original DreamerV2 Paper | Current Implementation | Notes |
| --- | --- | --- | --- |
| **Imagination horizon** | 15 steps | **15 steps** | Matches paper. Previous runs used 25; corrected. |
| **Actor gradient** | Dynamics backprop (continuous) | **REINFORCE** | Discrete `OneHotCategorical` gives zero dynamics grad without Gumbel. REINFORCE is correct for this action type. |
| **Actor entropy** | 1e-4 | **3e-3** | Crafter config override. Default 1e-3 caused policy collapse under sparse rewards; 3e-3 is the config.yaml value used in the DreamerV2 Crafter experiments. |
| **Actor LR** | 8e-5 | **1e-4** | Crafter config. |
| **Critic LR** | 2e-4 | **1e-4** | Crafter config. |
| **Model LR** | 3e-4 | **1e-4** | Crafter config; prevents over-fitting on early sparse rollouts. |
| **Gamma** | 0.99 | **0.999** | Crafter config; sparse achievement chains need longer effective horizon. |
| **Decoder output** | Sigmoid mean, small σ | **Sigmoid + σ=0.1 (fixed)** | Matches paper intent. Earlier runs had no sigmoid and σ=1.0, causing silent world-model failure. |
| **Reward predictor** | Discrete categorical bins (DreamerV3) | **MSE / Normal(μ, σ=1.0)** | Simpler than categorical; fixed σ avoids learned-σ collapse on sparse rewards. |
| **Discount predictor** | Binary BCE, `pred_discount=True` | **Implemented (binary BCE)** | Predicts P(episode continues). Used to dampen lambda-returns at imagined episode ends. Previously missing. |
| **Free nats** | 1.0 | **1.0** | Restored. Was incorrectly set to 0.0 in an earlier commit. |
| **KL balance** | 0.8 | **0.8** | Matches paper. Separate free-nats floors on lhs and rhs. |
| **Discrete latent** | 32 × 32 = 1024D | **32 × 32 = 1024D** | Identical. |
| **Recurrent state** | GRU, 1024D | **GRU, 1024D** | Identical. |
| **Encoder** | ResNet-style | **4-layer CNN (32→64→128→256)** | Lighter encoder sufficient for Crafter's 64×64 pixel art. |
| **Target critic** | Soft EMA (τ=0.01) | **Soft EMA (τ=0.01)** | Matches paper. Lambda-returns bootstrapped from target; online critic regresses targets. |
| **Replay buffer** | Uniform | **Prioritized (α=0.6, β=0.4)** | Extension beyond paper. Prioritizes surprising transitions. |
| **ε-greedy exploration** | Not in paper | **10% floor during training** | Extra exploration guard against actor collapse between world-model updates. |
| **Training interval** | Every env step | **Every 5 env steps** | Batching efficiency on CPU/MPS. |
| **Actor / critic size** | Larger (Atari) | **4-layer × 400 hidden** | Scaled down for Crafter's simpler dynamics. |

### What Was Corrected Since Earlier Runs

Three structural bugs caused the "silent world model" failure documented in TRAINING_NOTES.md v2:

1. **Decoder had no sigmoid and σ=1.0** — the mean was unbounded and log-prob was dominated by a constant term, so gradients through reconstruction were near zero. Fixed: `sigmoid(x)` for mean, `σ=0.1`.
2. **free_nats=0.0** — removed the pressure on the encoder to encode information when reconstruction signal was weak. The posterior collapsed to match the prior trivially. Fixed: `free_nats=1.0`.
3. **actor_grad='dynamics'** — gives zero gradients for discrete actions. Fixed: switched to REINFORCE with `actor_entropy=3e-3`.

### Training Runs and Checkpoints

**Legacy run (archived):** `archive/run_20260218_222805/`

Used `actor_grad='dynamics'` (broken gradient throughout). Checkpoints at 90k, 96k, 150k, 200k, 230k, 290k steps. The actor was never trained — uniform policy throughout. These episodes are still useful for visualization demos because the varied random actions produce non-trivial trajectories and the encoder embeddings do vary with observation.

**Fresh runs A/B/C (archived, collapsed):** `data/checkpoints/ckpt_270000`, `ckpt_470000`, `Fresh Checkpoints/ckpt_500000`

Switched to REINFORCE but had the decoder and free_nats bugs. Actor entropy collapsed to zero within ~5k gradient updates. World model never produced meaningful reconstructions (`obs_loss` ≈ 11,343 baseline throughout).

**Current config (pending new run):** All structural fixes are in code. The next run will serve as empirical validation. Success criteria: `obs_loss` drops below 9,000 within 50k steps; reconstruction PNGs show recognizable frames; `actor/entropy` stays above 0.5 through 50k updates.

**Evaluation tooling:**

```bash
# Run 50 episodes with a specific checkpoint step
python scripts/run_eval.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --checkpoint-step 660000 \
    --num-episodes 50

# Rank results by achievement diversity + reward
python scripts/find_best_episode.py --eval-dir data/eval

# Visual world-model sanity check
python generate_recon.py --checkpoint-dir data/checkpoints/ckpt_750000
```

### Data Availability

**In the git repo (tracked):**

- All Python source (`dreamer/`, `vis/`, `scripts/`)
- Documentation (`docs/`)
- `requirements.txt`, `scripts/requirements_eval.txt`
- Demo video (`data/Gameplay_Video.mp4`, `docs/video.gif`)

**Not in the repo (gitignored):**

- `archive/` (~11 GB) — legacy checkpoints, training logs, episode CSVs, videos
- `data/checkpoints/` — TF checkpoint files
- All `.csv`, `.mp4`, `.npz` files

Someone cloning the repo can run training from scratch. For the visualization tool to work out of the box, they need at minimum a few episode CSV+MP4 pairs and — for attribution columns — episodes produced by `run_eval.py` with a trained checkpoint.

---

## Part 1: Architectural Issues — Current Status

### Issue 1: Temporal Misalignment (RESOLVED for new eval data)

**Previous approach:** `data_manager.py` resampled ~200 sparse training-attribution samples uniformly across ~160 episode steps. This was a statistical approximation, not true per-step attribution.

**Current status:** Episodes produced by `scripts/run_eval.py` include attribution columns (`action_probability`, `value_estimate`, `exploration_bonus`, `world_model_score`) directly per step via `run_episode → policy.decision_attribution`. No resampling is performed. The resampling fallback remains only for legacy `event_log_*.csv` files that lack attribution columns.

**Residual limitation:** Training-time `decision_attribution.csv` is still sparse (~1k-step intervals). This file is only used for training health monitoring; it is not used in the visualization for eval episodes.

---

### Issue 2: Sparse Decision Logging During Training (PARTIALLY RESOLVED)

**Previous:** ~200 samples for 250k training steps (0.08% coverage), first environment only.

**Current:** The same 1k-step interval and first-env-only restriction applies to `decision_attribution.csv` during training. However, because eval CSVs now carry per-step attribution, the sparse training log is no longer the primary source for visualization. It remains useful only as a training health monitor (watching for entropy collapse and `obs_loss` stagnation).

**Still open:** A full action-probability histogram logged every 5k steps would surface actor collapse much earlier than the per-step taken-action probability. This is a low-effort instrumentation add.

---

### Issue 3: Episode-Level Attribution — RESOLVED

**Previous:** No causal link between training-phase decisions and episode outcomes. Decisions had to be resampled and approximated.

**Current:** `run_episode` calls `policy.decision_attribution(obs)` at every timestep and writes the result into the episode CSV row. The link is direct: each row records both what happened (action, reward, inventory, achievements) and what the policy believed at that exact step (value, probability, entropy, world-model score). True per-step causal association is now available for all eval episodes.

---

### Issue 4: Limited Step-Level Explainability

**Previous:** Raw numbers with no explanation of what in the observation triggered them.

**Current:** `vis/explainer.py` generates deterministic natural-language text per step from the attribution row, covering:

- Action choice with confidence label (high / moderate / low, from `action_probability`)
- Value trend (rising / stable / falling, from delta between consecutive `value_estimate` values)
- Reward context (significant / small / penalty)
- World-model confidence text (Dreamer) or entropy/advantage text (PPO)
- Vital-stat warnings (health, food, drink, energy drops of ≥2 in one step)
- Achievement unlock announcements

**Still open:** The explanation is template-driven and deterministic. It describes attribution signals correctly but does not explain *why* a visual feature (e.g., a tree nearby) drove a particular action. That requires either attention maps or LLM-generated causal narration. See Part 2, Improvement 1.

---

### Issue 5: Semantic Event Detection — Rule-Based (Unchanged)

`vis/SemanticEventDetector.py` uses hand-coded rules:

- `collect_` prefix on reward components → `resource_collected`
- `make_` prefix → `item_crafted`
- `place_` prefix → `structure_built`
- `defeat_` prefix → `enemy_defeated`
- `wake_up` → `survival`
- First movement action in a consecutive block → `exploration`

**Status:** The detector class is fully implemented, but as of this revision **it is not yet wired into the application** — nothing in `vis/` imports `SemanticEventDetector` or calls `detect_events()`, so its output does not currently feed the explanation panel or any plot. It is ready to integrate (a clean follow-up is to call it on episode load and use its events to populate the `achievement_unlocked` field the explainer already reads). The rules themselves are observation-derived (not arbitrary numeric cutoffs), so brittleness is low. LLM-based discovery of novel event types remains a future extension (see Part 2, Improvement 3).

---

### Issue 6: No Comparative Analysis

**Status:** Unchanged. The UI loads one episode at a time. `find_best_episode.py` provides batch ranking by achievement diversity and reward, but in-UI cross-episode comparison (e.g., "why did episode 5 outperform episode 12?") is not implemented. This is a future extension.

---

### Issue 7: Single-Machine UI

**Status:** Unchanged. All data is loaded into memory per episode. Sufficient for the study use case (dozens of episodes). A batch-analysis backend would be needed for thousands of episodes.

---

## Part 2: LLM-Powered Improvements

The improvements below are proposed extensions. The deterministic explanation layer in `vis/explainer.py` is already implemented and serves the current study. LLM layers would add higher-level narration and comparative capabilities.

### Improvement 1: LLM-Generated Episode Narratives

Produces a 3-sentence episode summary from the event CSV and attribution signals, describing what the agent tried to accomplish, key decisions, and exploration vs. exploitation balance. Would sit above the per-step explanation panel as an episode-level header.

### Improvement 2: Decision Explanation on Hover

When a user hovers over a decision-attribution plot point, generates a 1–2 sentence natural-language explanation of why that attribution pattern makes sense (or doesn't), incorporating visual context features extracted from the corresponding video frame.

### Improvement 3: Automatic Semantic Event Discovery

Replaces the hand-coded `SemanticEventDetector` rules with LLM inference across multiple episodes — discovering recurring behavioral patterns (e.g., "agent always collects wood before exploring stone") without requiring explicit threshold coding.

### Improvement 4: Comparative Episode Analysis

Allows natural-language questions across episodes: "Why did episodes with iron pickaxes have higher rewards?" The LLM receives episode summaries and attribution statistics for the queried group and returns an explanation.

### Improvement 5: Interactive Episode Querying

Semantic search over episode summaries using embeddings, then LLM aggregation. Supports questions like "Which episodes show the agent taking risks?" or "When does the exploration bonus spike?"

### Improvement 6: Real-Time Training Narrative

Every 50k training steps, generates a paragraph describing what the agent has learned — useful for monitoring long runs without reading raw metric tables.

---

## Part 3: Priority Ranking (Updated)

| Issue | Severity | Engineering or LLM | Status |
| --- | --- | --- | --- |
| Temporal misalignment | Critical | Engineering | **Resolved** (per-step attribution in eval CSVs) |
| Sparse decision logging | Critical | Engineering | **Partially resolved** (eval per-step; training still sparse) |
| Per-episode attribution loss | High | Engineering | **Resolved** |
| Step-level explainability | Medium | Deterministic templates | **Implemented** (`vis/explainer.py`) |
| Episode-level narration | Medium | LLM | Proposed extension |
| Semantic event detection | Medium | Rule-based (done) / LLM (extension) | Rule-based implemented |
| Comparative analysis | Medium | LLM | Proposed extension |
| Action probability histogram logging | Low | Engineering | Open (low effort) |
| Single-machine bottleneck | Low | Infrastructure | Not started |

---

## Recommended Implementation Order (Revised)

Steps 1–3 from the original order are **complete**. Remaining work:

1. Add action-probability histogram to training logs (low effort, closes the last training-visibility gap)
2. Validate world-model fix with a new ≥100k-step run; inspect recon PNGs at 10k, 20k, 30k steps
3. If training is working: integrate LLM-generated episode narratives as an optional above-the-explanation-panel header
4. Add on-hover decision explanation (requires frame → visual feature extraction step)
5. Implement semantic event discovery for novel environments
6. Build comparative analyzer (requires batch infrastructure beyond single-machine UI)

---

## Summary: How LLMs Add Value

| Feature | Without LLM | With LLM |
| --- | --- | --- |
| Understanding a step | "value=0.087, explore=1.93, prob=0.23" | "Agent acted with moderate confidence; value is rising, suggesting it expects progress soon." (now: deterministic templates) |
| Understanding an episode | Read row-by-row | "The agent explored early, found wood, then exploited crafting chains." (proposed) |
| Finding patterns | Manual code + thresholds | Automatic discovery across hundreds of episodes (proposed) |
| Comparing episodes | Read each individually | "These 3 episodes failed because the agent never built a furnace." (proposed) |
| Debugging training | Stare at reward curves | "Step 150k: agent shifted from exploration to exploitation." (proposed) |

**Key insight:** The deterministic template layer already translates numbers into readable sentences at the step level. LLMs extend this to the episode level and above, where the combinatorial space of possible narratives exceeds what templates can cover.
