# Crafter / DreamerV2 Training Audit (v3)

**Date compiled:** 2026-04-26
**Scope:** All commits from `7b800ff` through HEAD, checkpoints in `data/checkpoints/`, and the current implementation of `dreamer/core.py`, `dreamer/policy.py`, `dreamer/train.py`, and `scripts/run_eval.py`.

**v2 of this report (2026-04-18)** identified that all three "fresh" runs (270k, 470k, 500k) collapsed because the world model was silent — the decoder never learned to reconstruct frames, so the imagined returns driving the actor and critic were pure noise.

**This v3** documents the parameter and code changes merged since v2, explains which issues from the v2 recommendations have been closed, and updates the status of the training-fix track.

---

## TL;DR (v3)

1. **All structural fixes from the v2 recommendation list are now in code.** The decoder uses sigmoid + fixed σ=0.1, free_nats=1.0 is restored, the discount predictor is implemented, REINFORCE is confirmed as the actor gradient, actor_entropy=3e-3, ε-greedy 10% floor is active during training.
2. **The decoder reconstruction smoke test is automated.** `train.py` saves a side-by-side original/reconstruction PNG every 10k steps to the log directory. `generate_recon.py` is also available as a standalone checker.
3. **Step-gated CSV logging is fixed.** `dreamer_training_log.csv` now writes even when no episode has completed that interval, eliminating the "short run = no log" blind spot from v2.
4. **Eval-to-attribution linking is now per-step.** `scripts/run_eval.py` writes attribution columns (action_probability, value_estimate, exploration_bonus, world_model_score) directly into each episode's CSV via `run_episode`. The resampling workaround in the old architecture is no longer needed for eval data.
5. **Checkpoint selection is now flexible.** `run_eval.py` accepts `--checkpoint-step` to target any specific step inside a checkpoint folder, and `find_best_episode.py` ranks results by achievement diversity and reward.
6. **Whether these changes fix the world-model silence requires a new training run** to verify empirically. The parameters are aligned with the DreamerV2/Crafter paper config; the next step is running ≥100k steps and inspecting the saved reconstruction PNGs and tensorboard scalars.

---

## 1. Updated run timeline

| Run | Checkpoint dir | Steps | Status | Config |
| --- | --- | --- | --- | --- |
| Legacy 0→1.3M | `data/checkpoints/Old Checkpoints/ckpt_*` | ~1.3M | Archived. Not touched since `7b800ff`. | `actor_grad='dynamics'` (broken gradient). Uniform policy throughout. |
| Fresh A | `data/checkpoints/ckpt_270000` | 0→270k | Archived. Collapsed. | REINFORCE enabled but `free_nats=0`, no sigmoid decoder, actor_entropy=1e-4. Entropy collapsed within ~5k updates. |
| Fresh B | `data/checkpoints/ckpt_470000` | 0→470k | Archived. Collapsed. | Same as A (parallel run). |
| Fresh C | `data/checkpoints/Fresh Checkpoints/ckpt_500000` | 0→500k | Archived. Collapsed. | lr=8e-5, actor_entropy=1e-3. Delayed collapse but same failure mode. |
| **Current config** | `data/checkpoints/ckpt_<N>` | TBD | **Parameters ready; next run pending.** | sigmoid decoder + σ=0.1, free_nats=1.0, actor_entropy=3e-3, actor_lr=1e-4, γ=0.999, discount predictor, ε-greedy 10%. |

---

## 2. What changed since v2 (per recommendation §6.2)

### 2.1 Fix 1 — Decoder output range (DONE)

**v2 identified:** The decoder mean was unbounded and σ=1.0, making log-prob dominated by a constant term. Gradients through `(x − μ)²/2` were weak, so the encoder had no learning signal.

**Current code (`dreamer/core.py`):**

```python
mean = tf.sigmoid(x)   # constrains mean to (0,1)
return tfd.Independent(tfd.Normal(mean, 0.1), 3)
```

`σ=0.1` tightens the reconstruction loss so gradients are meaningful even for small errors. The sigmoid ensures the mean lives in the same normalized [0,1] range as the target observations (`obs/255`).

### 2.2 Fix 2 — Reward head std collapse (DONE)

**v2 identified:** With a learned σ, the reward head's σ grew unbounded on sparse Crafter rewards, killing the mean's gradient.

**Current code:**

```python
self.reward_predictor = DenseDecoder(
    output_shape=[1],
    dist="mse",   # fixed σ=1.0 per paper
)
```

### 2.3 Fix 3 — free_nats restored to 1.0 (DONE)

**v2 identified:** `free_nats=0.0` (set in `7b800ff`) let the posterior trivially match the prior when reconstruction signal was weak, causing KL to stay near zero. The DreamerV2 and Crafter configs both use 1.0.

**Current code:**

```python
free_nats=1.0   # DreamerV2/Crafter default
```

The free-nats floor is applied separately to both the lhs and rhs of the balanced KL:

```python
kl_lhs = tf.maximum(kl_lhs, self.free_nats)
kl_rhs = tf.maximum(kl_rhs, self.free_nats)
```

### 2.4 Fix 4 — Actor gradient: REINFORCE confirmed (DONE)

**v2 identified:** `actor_grad='dynamics'` gives zero gradients for discrete `OneHotCategorical` samples without a straight-through Gumbel estimator. `7b800ff` switched to REINFORCE.

**Current code:**

```python
actor_grad='reinforce'   # discrete actions → REINFORCE
# Actor loss: -E[log π(a|s) · stop_grad(returns - values)]
advantage = tf.stop_gradient(returns - values)
actor_loss = -tf.reduce_mean(log_probs * advantage)
actor_loss -= self.actor_entropy * tf.reduce_mean(entropies)
```

### 2.5 Fix 5 — actor_entropy raised to 3e-3 (DONE)

**v2 identified:** 1e-3 is too weak; REINFORCE sharpens logits faster than the entropy bonus can push back, causing collapse within ~5k updates.

**Current code:**

```python
actor_entropy=3e-3   # Crafter override
```

### 2.6 Fix 6 — ε-greedy exploration floor (DONE)

**v2 identified (§6.2 item 4 alternative):** Adding a hard exploration floor keeps the replay buffer diverse even after actor logits start sharpening.

**Current code (`dreamer/policy.py`):**

```python
def __call__(self, obs, expl_amount=0.1):
    ...
    if self.training and expl_amount > 0 and np.random.random() < expl_amount:
        action_np = int(np.random.randint(0, self.agent.action_size))
```

10% of training steps take a uniformly random action regardless of actor output.

### 2.7 Fix 7 — Discount (continue) predictor (DONE)

**v2 identified (via architecture analysis):** Without a discount predictor, imagined trajectories don't account for episode termination. Lambda-returns bootstrap past hard horizon cutoffs.

**Current code:**

```python
self.discount_predictor = DenseDecoder(
    output_shape=[1], dist="binary",
)
# In imagination: effective discount = gamma * P(episode continues)
discount = self.discount_predictor(model_features).mean()
# In compute_return: disc = gamma * discount_predictions
```

### 2.8 Fix 8 — Target critic with soft EMA (DONE)

A slow-updating target critic prevents the critic from chasing its own tail. Lambda-returns are bootstrapped from the target; the online critic regresses those targets.

**Current code:**

```python
tau = 0.01
for w_target, w_online in zip(
    self.target_critic.trainable_variables,
    self.critic.trainable_variables,
):
    w_target.assign(w_target * (1.0 - tau) + w_online * tau)
```

### 2.9 Fix 9 — Decoder reconstruction smoke test (DONE)

**v2 §3.1 identified:** "Add a smoke test at the start of any new run: decode ~10 observations and visually compare to originals."

**Current code (`dreamer/train.py`):**

```python
if step_count % 10000 < num_envs and hasattr(policy, 'agent'):
    # encoder → RSSM.observe → decoder.mean → side-by-side PNG
    imageio.imwrite(os.path.join(log_dir, f'recon_{step_count:07d}.png'), ...)
```

`generate_recon.py` also provides a standalone one-shot checker:

```bash
python generate_recon.py --checkpoint-dir data/checkpoints/ckpt_750000
```

If the reconstruction PNG shows uniform gray at 10k steps, the world model has the same silent failure as v2. Stop and diagnose before continuing.

### 2.10 Fix 10 — Step-gated training CSV (DONE)

**v2 §6.3 identified:** `dreamer_training_log.csv` was episode-gated, so short runs with long episodes produced no log rows.

**Current code:**

```python
# Writes even if no episode has completed (avg_reward = nan in that case)
metrics_dict = {'step': step_count, 'avg_reward': avg_reward, 'avg_length': avg_length}
metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
```

---

## 3. What is still open from v2

### 3.1 TensorBoard step alignment

**v2 §6.3:** `tf.summary` scalars all used `step=1` (writer bug). `global_step` variable exists but needs to be threaded to the writer's default step.

**Status:** The `log_metrics` method now uses `int(gs.numpy())` as the step, which is correct. However, a step-aligned default step via `tf.summary.experimental.set_step(global_step_var)` inside `@tf.function` traced code would be cleaner. Low priority now that most attribution logging happens in eval CSVs rather than tensorboard.

### 3.2 Action probability histogram logging

**v2 §6.3:** Log full action distributions (not just the one action taken), so actor collapse is visible earlier.

**Status:** Not yet implemented. The current `decision_attribution.csv` logs only the taken action's probability. Adding `np.histogram` on the softmax vector to the training loop would expose collapse within the first 5k steps rather than requiring a post-hoc tensorboard inspection.

### 3.3 Empirical validation of the world-model fix

**Status:** The parameters match the DreamerV2/Crafter paper config. Validation requires a new ≥100k-step run. Success criteria:

- `model/obs_loss` drops from ~11,343 to ≤9,000 within 50k steps
- Reconstruction PNGs at 10k, 20k, 30k are recognizably frame-like (not uniform gray)
- `actor/entropy` stays above 0.5 through 50k updates
- `action_probability` in the attribution log shows a non-trivial distribution (not all 1.0)

---

## 4. Evaluation tooling updates (new since v2)

### 4.1 `scripts/run_eval.py` — checkpoint step selection

`run_eval.py` now accepts `--checkpoint-step` to target a specific step within a checkpoint folder, rather than always using the highest-step file.

```bash
# Use highest step in folder:
python scripts/run_eval.py --checkpoint-dir data/checkpoints/ckpt_750000 --num-episodes 50

# Target a specific step:
python scripts/run_eval.py --checkpoint-dir data/checkpoints/ckpt_750000 \
    --checkpoint-step 660000 --num-episodes 50
```

This is especially useful when a training folder holds checkpoints at multiple steps (the `CheckpointManager` keeps up to 10).

### 4.2 `scripts/find_best_episode.py` — episode ranker

Scans all `stats.jsonl` files produced by `crafter.Recorder` and ranks episodes by unique achievement types (primary) and total reward (secondary).

```bash
python scripts/find_best_episode.py --eval-dir data/eval --top 10
```

### 4.3 Attribution is now per-step in eval CSVs

The v2 architecture relied on resampling sparse training attribution (~200 samples per run) across episode steps. That workaround is now only needed for legacy archived episodes. Eval episodes produced by `run_eval.py` include attribution columns directly per step (via `run_episode` → `policy.decision_attribution`):

```text
logit, action_probability, value_estimate, exploration_bonus, world_model_score
```

The visualizer reads these directly from the episode CSV without any resampling step.

---

## 5. Hyperparameter summary (current)

| Parameter | Current value | Paper default | Why different |
| --- | --- | --- | --- |
| `actor_entropy` | 3e-3 | 1e-3 | Crafter config; sparse rewards need stronger entropy push. |
| `actor_grad` | `'reinforce'` | `'dynamics'` (continuous) | Discrete `OneHotCategorical` → zero dynamics grad without Gumbel. |
| `actor_lr` | 1e-4 | 8e-5 | Crafter config. |
| `critic_lr` | 1e-4 | 2e-4 | Crafter config. |
| `model_lr` | 1e-4 | 3e-4 | Crafter config; prevents over-fitting on early sparse rollouts. |
| `gamma` | 0.999 | 0.99 | Crafter config; long achievement chains need long horizon. |
| `free_nats` | 1.0 | 1.0 | Restored from the incorrect 0.0 in `7b800ff`. |
| `kl_balance` | 0.8 | 0.8 | Matches paper. |
| `imagination_horizon` | 15 | 15 | Matches paper. |
| `decoder σ` | 0.1 (fixed) | ~0.1 (DreamerV3) | Tightens reconstruction; prevents log-prob constant dominating. |
| `reward head` | MSE (σ=1.0) | Categorical bins (DV3) | Simpler; avoids learned-σ collapse on sparse rewards. |
| `discount predictor` | Binary BCE | Binary BCE | Matches paper. |
| `ε-greedy` | 10% | Not in paper | Additional exploration floor on top of entropy bonus. |
| `training_interval` | 5 env steps | 1 | Batch efficiency on CPU/MPS. |
| `replay buffer` | Prioritized (α=0.6) | Uniform | Extra; not needed for correctness but helps in sparse-reward regime. |

---

## 6. Decision for the study session

If running the study before a new training run completes, the recommendations from v2 §6.1 still apply:

- Use the best eval episode from an existing checkpoint (rank with `find_best_episode.py`).
- The new per-step attribution in eval CSVs gives genuine per-step variety in `value_estimate` and `world_model_score` even from a partially-trained agent — this is better demo material than the legacy resampled attribution.
- Frame the UI as "here is how the explanation system works on a mid-training agent" — that is an accurate and honest description.

---

## 7. Files referenced

- Current parameter source: `dreamer/core.py:375–452` (`class DreamerV2` at 375; `__init__` defaults span ~378–452)
- Training loop with recon + step-gated CSV: `dreamer/train.py:148–387`
- ε-greedy floor: `dreamer/policy.py:311–353`
- Eval runner with checkpoint-step arg: `scripts/run_eval.py`
- Episode ranker: `scripts/find_best_episode.py`
- Standalone recon checker: `generate_recon.py`
- Archived tensorboard events (v2 analysis still valid):
  - `data/checkpoints/ckpt_270000/tensorboard/20260416-212448/`
  - `data/checkpoints/ckpt_470000/tensorboard/20260417-035115/`
  - `data/checkpoints/Fresh Checkpoints/ckpt_500000/tensorboard/20260417-100047/`
