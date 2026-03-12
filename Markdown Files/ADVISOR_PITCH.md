# Advisor Pitch: Crafter RL Analysis Platform

## The 5-Minute Explanation

### Problem Statement
When training reinforcement learning agents, we get **what happened** (episodes) but not **why it happened** (agent's reasoning). Understanding the agent's decision-making process requires:
1. Visualizing complex temporal patterns (rewards, resources, achievements)
2. Correlating agent beliefs (value, exploration) with actual outcomes  
3. Reconstructing the causal chain: observation → belief → decision → outcome

### What We Built
A **full-stack platform** that trains DreamerV2 agents on Crafter (3D open-world game with ~40 tasks) and provides **interpretable decision analysis**:

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
- **Innovation**: Resample training decision data uniformly across episodes
- Handles the temporal mismatch: training steps (90K+) ≠ episode steps (0-300)
- Merges episode outcomes with policy beliefs for synchronized analysis

**3. Visualization UI** (`vis/widgets.py` + `vis/main.py`)
- PyQt5 + pyqtgraph for interactive plots
- **3 synchronized panels**:
  - Video + timeline slider (frame-step alignment)
  - Reward metrics: cumulative rewards + component breakdown (stacked areas)
  - Decision attribution: value estimate, action probability, exploration bonus, world-model confidence
- Click on reward events to inspect agent's mental state at that moment

### What Makes It Novel

1. **Temporal Alignment**: Episodes and training signals on different scales unified via resampling
2. **Multi-modal Integration**: Video + CSV data + metrics synchronized
3. **Interpretable Signals**: Shows agent's beliefs (not just actions) during decision-making
4. **Achievement Tracking**: Maps low-level state (inventory) to high-level milestones (crafted iron pickaxe?)

### Learning Outcomes

**For RL practitioners:**
- How does exploration evolve? (exploration_bonus trends)
- Are high-value states predictive of success? (value vs reward correlation)
- Do policy actions align with beliefs? (action_prob vs success rate)

**For interpretability researchers:**
- Demonstrates multimodal decision attribution (visual + temporal + semantic)
- Shows how to extract and visualize latent model signals
- Reveals limitations of policy explanation at scale

### Current Metrics  
- Training: 250K+ steps on 4 parallel environments
- Episodes: 500+ recorded episodes with synchronized decision attribution
- Viz: Real-time video-metric sync with <50ms latency

---

## One-Slide Summary

> We built a **DreamerV2 agent** that plays Crafter, trained for 250K+ steps on 4 parallel envs with prioritized replay. We **record decision internals** (value, action probability, exploration) every ~1000 training steps and **resample them to align with episodes** for analysis. A PyQt5 interface shows **video + synchronized metrics + decision attribution plots**, letting researchers correlate agent beliefs with outcomes and understand exploration/value trade-offs over time.

---

## FAQ for Advisor Questions

**Q: How do you handle the temporal mismatch (training vs episode)?**  
A: Uniformly resample ~200 training decision samples across ~160 episode steps using linear interpolation. Assumption: policy evolution is roughly smooth.

**Q: Why not log decisions per-episode?**  
A: Training runs for 250K+ steps across thousands of episodes. Per-episode logging would be prohibitive (~500K logs). Current interval (~1K steps) is a tradeoff.

**Q: What's the scientific contribution?**  
A: (1) Multimodal interpretability framework, (2) Temporal alignment technique for decision attribution, (3) Empirical analysis showing exploration-value dynamics.

**Q: Limitations?**  
A: Resampling loses fine-grained temporal resolution; only Dreamer signals (no counterfactuals/ablations); single-algorithm analysis.

---
