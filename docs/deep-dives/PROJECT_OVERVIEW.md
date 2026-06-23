# Advisor Pitch: Crafter RL Analysis Platform

**Last updated:** 2026-06-22

## The 5-Minute Explanation

### Problem Statement
When training reinforcement learning agents, we get **what happened** (episodes) but not **why it happened** (agent's reasoning). Understanding the agent's decision-making process requires:
1. Visualizing complex temporal patterns (rewards, resources, achievements)
2. Correlating agent beliefs (value, exploration) with actual outcomes  
3. Reconstructing the causal chain: observation → belief → decision → outcome

### What We Built
A **full-stack platform** that trains DreamerV2 agents on Crafter (a 2D open-world survival game with 22 achievements) and provides **interpretable decision analysis**:

```
Training Phase                  Analysis Phase
─────────────────              ──────────────────
DreamerV2 Agent ──────────────→ Visualization ──→ Insights
  (policy learns behavior)       (policy internals exposed)
      │                              │
      ├→ Episode Logs               ├→ Video Playback
      │  (what happened)            │  (synchronized frame)
      │
      └→ Decision Attribution       └→ Decision Attribution Plot
         (what agent believed)         (value, probability, exploration)
```

### Key Components

**1. Training Pipeline** (`dreamer/train.py`)
- 4 parallel environments with prioritized replay
- DreamerV2 core: recurrent state-space model with imagination
- Every ~1000 training steps: log policy decisions (action prob, value, exploration)

**2. Data Management** (`vis/data_manager.py`)
- **Current approach**: eval episodes carry per-step decision attribution directly in their CSV (written by `scripts/run_eval.py` → `run_episode` → `policy.decision_attribution`), so each row records both what happened and what the policy believed at that exact step — no approximation
- **Legacy fallback**: for older `event_log_*.csv` files that lack attribution columns, `data_manager.py` can still resample a separate sparse `decision_attribution.csv` across the episode length (the original temporal-alignment technique)
- Merges episode outcomes with policy beliefs for synchronized analysis

**3. Visualization UI** (`vis/widgets.py` + `vis/main.py`)
- PyQt5 + pyqtgraph for interactive plots
- **3 synchronized panels**:
  - Video + timeline slider (frame-step alignment)
  - Reward metrics: cumulative rewards + component breakdown (stacked areas)
  - Decision attribution: value estimate, action probability, exploration bonus, world-model confidence
- A deterministic explanation panel that narrates each step in natural language
- Click on reward events to inspect agent's mental state at that moment

### What Makes It Novel

1. **Per-step decision attribution**: eval episodes record the policy's beliefs (value, action probability, exploration bonus, world-model score) at every timestep, computed from the actual checkpoint used in the rollout — no resampling or interpolation (the earlier resampling technique remains only as a legacy-data fallback)
2. **Multi-modal Integration**: Video + CSV data + metrics synchronized
3. **Interpretable Signals**: Shows agent's beliefs (not just actions) during decision-making
4. **Achievement Tracking**: Maps low-level state (inventory) to high-level milestones (crafted iron pickaxe?)
5. **Template-based Narration**: Turns decision attribution into readable explanations without relying on an external LLM

### Learning Outcomes

**For RL practitioners:**
- How does exploration evolve? (exploration_bonus trends)
- Are high-value states predictive of success? (value vs reward correlation)
- Do policy actions align with beliefs? (action_prob vs success rate)

**For interpretability researchers:**
- Demonstrates multimodal decision attribution (visual + temporal + semantic)
- Shows how to extract and visualize latent model signals
- Reveals limitations of policy explanation at scale

### Platform Capabilities
- Training: DreamerV2 on 4 parallel environments, step-suffixed checkpoint/log folders, periodic reconstruction snapshots
- Evaluation: batch episode rollout with per-step decision attribution, achievement-diversity ranking (`find_best_episode.py`)
- Viz: synchronized video ↔ timeline ↔ metric playback with deterministic per-step explanations

> **Note on training status:** the world-model fixes (sigmoid decoder, `free_nats=1.0`, REINFORCE actor, discount predictor) are all in code, but the prior "fresh" runs (270k/470k/500k) collapsed before those fixes landed. A new validation run is the immediate next step — see `TRAINING_NOTES.md` for the empirical success criteria. Demo episodes are drawn from existing checkpoints and are honestly framed as a mid-training agent.

---

## One-Slide Summary

> We built a **DreamerV2 agent** that plays Crafter (2D survival, 22 achievements) on 4 parallel envs with prioritized replay. We **record decision internals per step** (value, action probability, exploration bonus, world-model score) directly in each eval episode's CSV via the actual checkpoint used in the rollout — no resampling (a resampling fallback remains only for legacy logs). A PyQt5 interface shows **video + synchronized metrics + decision attribution plots + a deterministic explanation panel**, letting researchers correlate agent beliefs with outcomes and understand exploration/value trade-offs over time. The visualizer also handles PPO-style columns when they appear in loaded CSVs.

---

## FAQ for Advisor Questions

**Q: How do you handle the temporal mismatch (training vs episode)?**  
A: We no longer rely on resampling for analysis. Eval episodes are run from a fixed checkpoint and attribution is computed per step inside `run_episode`, so the policy beliefs are already aligned to episode timesteps. The original resampling technique (uniformly mapping ~200 sparse training samples across episode steps via interpolation) remains only as a fallback for legacy logs that lack per-step attribution columns.

**Q: Why is training-time logging still sparse, then?**  
A: During training, `decision_attribution.csv` is sampled roughly every 1,000 steps (first env only) purely as a health monitor for actor collapse. It is intentionally not the analysis source — eval CSVs are. This keeps training cheap while giving analysis full per-step resolution.

**Q: What's the scientific contribution?**  
A: (1) Multimodal interpretability framework, (2) per-step decision attribution computed from the rollout checkpoint, (3) a deterministic, template-based explanation layer grounded in explanation-template theory (see `EXPLANATION_SYSTEM.md`).

**Q: Limitations?**  
A: The training pipeline is DreamerV2-only even though the visualization is algorithm-aware (PPO display-compatible); the explanation layer is template-based rather than causal; cross-episode comparison and the semantic event detector are not yet wired into the UI.

---
