# Dodge et al Paper on Explanation Templates (https://ceur-ws.org/Vol-2582/paper2.pdf) -

This paper is directly relevant to my work — and notably, it's **co-authored by Dr. Jonathan Dodge**, your advisor. Here's a breakdown of what it argues and exactly how it connects to the Crafter module:

***

## What the Paper Is About

The paper distinguishes two ways to evaluate XAI explanations: [ceur-ws](https://ceur-ws.org/Vol-3793/paper_57.pdf)

- **ExpS (Explanation Satisfaction)** — did the *user* feel the explanation helped them understand the AI? Measured via tasks like think-aloud or prediction tasks, after interacting with the system.
- **ExpG (Explanation Goodness)** — is the *explanation itself* good, independent of a user? Measured via checklists or by analyzing explanation templates across many inputs.

The core argument is: most XAI research only does ExpS, but ExpG — especially via **explanation templates** — is necessary to scale evaluation and catch problems (like self-refuting explanations) that individual user sessions will never surface.

***

## How It Directly Helps Your Crafter Module

### 1. NLP Explanation Templates — Text Panel Addition
The paper's central contribution is the concept of an **explanation template**: a structured sentence with variable slots filled in by the agent's data. You could add a **natural language explanation panel** to your interface that auto-generates text from your existing Decision Attribution signals. For example: [ceur-ws](https://ceur-ws.org/Vol-3793/paper_57.pdf)

> *"At step 47, the agent chose to **Move North** with **high confidence** (action probability: 0.87). Its **value estimate was rising**, suggesting it anticipated a reward. The world-model score was **stable**, meaning the outcome matched its expectations."*

The template structure would be:
```
"At step {t}, the agent chose to {action} with {confidence_level} confidence 
(action probability: {prob}). Its value estimate was {rising/falling}, 
{reward_anticipation}. The world-model score was {stable/dropping}, 
meaning {outcome_match}."
```

You already have all the variables — `action`, `action_probability`, `value_estimate`, `world_model_score` — in your CSV data. An LLM could either fill the template dynamically or generate free-form explanations conditioned on these signals. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82724951/356cac81-3f2d-41a7-8c2e-583ef87a0c08/PROJECT_GUIDE.md)

***

### 2. Critical States — Highlight Pivotal Moments Automatically
The paper cites Huang et al.'s **criticality measure**  — selecting decision points where the agent perceived its choice to matter most. This maps precisely onto your Decision Attribution graph: steps where **value estimate is high AND action probability is high** are exactly the critical states. You could: [ceur-ws](https://ceur-ws.org/Vol-3793/paper_57.pdf)

- Auto-flag these steps with a marker on the Reward Timeline
- Snap the timeline to the highest-criticality moments during the study
- Surface these as "moments worth explaining" in the NLP panel

This would address a real problem the paper identifies: in a ~3-minute episode, participants are "gazing into a vast expanse through a tiny peephole" — surfacing critical states helps them focus.

***

### 3. Prediction Task — Add a Study Task to Your Interface
The paper discusses the **Prediction Task** as a key ExpS evaluation method: ask the participant to predict the agent's next action *before* revealing it. Your interface could support this by: [ceur-ws](https://ceur-ws.org/Vol-3793/paper_57.pdf)

- Pausing the video at a flagged critical state
- Hiding the next frame and asking the participant: *"What do you think the agent will do next?"*
- Revealing the actual action and comparing

This would give your user study a **quantitative metric** beyond just think-aloud commentary, which the paper notes is important but noisy.

***

### 4. Framing Your Study Design (ExpS vs. ExpG)
Your current study is an **ExpS study** — participants interact with the interface and you collect qualitative impressions. The paper directly warns about three pitfalls you should plan for: [ceur-ws](https://ceur-ws.org/Vol-3793/paper_57.pdf)

| Paper's Warning | Implication for Your Study |
|---|---|
| Limited exposure (only ~14 decisions seen in 2 hours) | Curate specific episode moments rather than having participants watch freely |
| Ceiling/floor effects in prediction accuracy | Choose episodes with a mix of obvious and ambiguous decisions |
| Individual variability moderates task performance | Collect demographic info on gaming experience and tech background |

***

### 5. ExpG Evaluation of Your Visualization Design
Beyond the user study, the paper argues you should also evaluate your *explanation templates themselves* using ExpG — checking for **soundness** (does the explanation say only true things?) and **completeness** (does it tell the whole story?). For your NLP panel, this means asking: does the generated text ever contradict the visual signal? Does it omit important context from the reward timeline?

***

## Bottom Line

The most concrete addition you can make is a **natural language explanation panel** driven by templates that consume your existing Decision Attribution CSV columns. This directly extends your visualization from showing signals to *narrating* them — which is the LLM/NLP angle you were looking for — and it's grounded in the exact theoretical framework your advisor has published. The Prediction Task integration would simultaneously strengthen your user study methodology in a way that directly cites Dodge & Burnett's own prior work.





# Integration Plan 1 -
Unfortunately, the repository appears to be **private or access-restricted** — all attempts to fetch the code directly are failing. I can still give you concrete, actionable guidance based on what I already know from your `PROJECT_GUIDE.md` and the screenshots. Here's the full plan:

***

## 1. XAI Template Integration

### Where to Add It — New File: `vis/explainer.py`

Create a dedicated module that reads the same CSV data your `VisDataManager` already loads and generates natural language explanations. Keep it separate from `widgets.py` to avoid coupling.

```python
# vis/explainer.py

CONFIDENCE_THRESHOLDS = {"high": 0.75, "medium": 0.5, "low": 0.0}
VALUE_CHANGE_THRESHOLD = 0.05

def confidence_label(prob):
    if prob >= 0.75: return "high confidence"
    if prob >= 0.5:  return "moderate confidence"
    return "low confidence / exploring"

def value_direction(prev, curr):
    delta = curr - prev
    if delta > VALUE_CHANGE_THRESHOLD:   return "rising", "anticipating a reward"
    if delta < -VALUE_CHANGE_THRESHOLD:  return "falling", "not expecting a payoff"
    return "stable", "neutral about the outcome"

def world_model_label(score):
    if score >= 0.75: return "stable", "the outcome matched its predictions"
    if score >= 0.4:  return "uncertain", "the outcome was somewhat surprising"
    return "low", "the agent was surprised — its world model did not predict this"

def generate_explanation(step, action, action_prob, value_curr, value_prev,
                         world_model_score, exploration_bonus,
                         achievement_unlocked=None, algorithm="dreamer"):
    conf = confidence_label(action_prob)
    vdir, vmeaning = value_direction(value_prev, value_curr)
    wm_label, wm_meaning = world_model_label(world_model_score)

    lines = [
        f"At step {step}, the agent chose to **{action}** with {conf} "
        f"(action probability: {action_prob:.2f}).",
        f"Its value estimate was {vdir}, meaning it was {vmeaning}.",
    ]

    if algorithm == "dreamer":
        lines.append(
            f"The world-model score was {wm_label}: {wm_meaning}."
        )
        if exploration_bonus > 0.7:
            lines.append(
                "The exploration bonus was elevated — "
                "the agent was in relatively unfamiliar territory."
            )
    elif algorithm == "ppo":
        lines.append(
            f"Entropy was {'high' if exploration_bonus > 0.6 else 'low'}, "
            f"suggesting the agent was {'exploring' if exploration_bonus > 0.6 else 'exploiting a known strategy'}."
        )

    if achievement_unlocked:
        lines.append(
            f"✅ This step unlocked the **{achievement_unlocked}** achievement."
        )

    return " ".join(lines)
```

***

### Where to Wire It — `vis/widgets.py`

In the `DecisionAttributionWidget` (or wherever the bottom graph lives), add a text box below the graph that calls `generate_explanation()` whenever the timeline cursor moves:

```python
# In your attribution widget's update method
from vis.explainer import generate_explanation

def on_step_changed(self, step_idx):
    row = self.data.iloc[step_idx]
    prev_value = self.data.iloc[max(0, step_idx - 1)]["value_estimate"]

    text = generate_explanation(
        step=step_idx,
        action=row.get("executed_action", row["action"]),
        action_prob=row.get("action_probability", 0.5),
        value_curr=row.get("value_estimate", 0.5),
        value_prev=prev_value,
        world_model_score=row.get("world_model_score", 0.5),
        exploration_bonus=row.get("exploration_bonus", 0.0),
        achievement_unlocked=row.get("achievement", None),
        algorithm=self.algorithm_mode  # "dreamer" or "ppo"
    )
    self.explanation_label.setText(text)
```

The `explanation_label` is a simple `QLabel` with word-wrap enabled, placed beneath the Decision Attribution graph — this is the visual "limelight" change you already wanted to make.

***

## 2. PPO Support — What Needs to Change

Your `PROJECT_GUIDE.md` already mentions that PPO columns (`entropy`, `advantage`) are supported in the CSV schema, but the interface label says "Dreamer" everywhere. Here's what needs updating: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/82724951/356cac81-3f2d-41a7-8c2e-583ef87a0c08/PROJECT_GUIDE.md)

### `vis/data_manager.py` — Algorithm Auto-Detection

```python
def detect_algorithm(df):
    has_dreamer = {"exploration_bonus", "world_model_score"}.issubset(df.columns)
    has_ppo     = {"entropy", "advantage"}.issubset(df.columns)
    if has_dreamer: return "dreamer"
    if has_ppo:     return "ppo"
    return "unknown"
```

Call this once on load and store it as `self.algorithm`. Pass it downstream to the widgets and the explainer.

### `vis/widgets.py` — Dynamic Panel Title & Signals

| Currently hardcoded | Should become |
|---|---|
| "Decision Attribution – Dreamer" | "Decision Attribution – {algorithm.upper()}" |
| Plots: `exploration_bonus`, `world_model_score` | If PPO: swap to `entropy`, `advantage` |
| Signal labels in legend | Conditionally show "Entropy" / "Advantage" vs "Exploration Bonus" / "World-Model Score" |

### `vis/main.py` — Status Bar or Header Badge

Add a small label (e.g., top-right of the right panel) that shows **"Agent: DreamerV2"** or **"Agent: PPO"** based on the detected algorithm. This helps study participants understand what they're looking at.

***

## 3. LLM Extension (Optional Upgrade)

Once the template-based explainer works, you can optionally route the template output through a local LLM (e.g., `ollama` with `llama3`) to make the language more natural:

```python
# Optional: in vis/explainer.py
def enrich_with_llm(template_text, model="llama3"):
    import requests
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": f"Rewrite this AI explanation in plain, friendly language for a non-expert: {template_text}",
        "stream": False
    })
    return resp.json().get("response", template_text)
```

This gracefully falls back to the raw template if the LLM is unavailable — important for a study setting where you don't want a network dependency breaking the interface mid-session.

***

## Suggested Integration Order

1. **Make the repo public** (or share files here) so the wiring can be verified against your actual code
2. Add `vis/explainer.py` with the template engine
3. Add `detect_algorithm()` to `data_manager.py`
4. Update the Decision Attribution widget title and signal selection dynamically
5. Add the `QLabel` explanation text box beneath the graph
6. Test with both a DreamerV2 CSV and a PPO CSV to confirm the branching works
7. (Optional) Plug in the LLM enrichment layer










# Integration Plan 2 - 

### New File: `vis/explainer.py`

Create this from scratch. It has no dependencies on your existing code — it just takes values and returns strings.

```python
# vis/explainer.py

ALGO_SIGNALS = {
    "dreamer": ["value_estimate", "action_probability",
                "exploration_bonus", "world_model_score"],
    "ppo":     ["value_estimate", "action_probability",
                "entropy", "advantage"],
}

def _confidence(prob):
    if prob >= 0.75: return "high confidence"
    if prob >= 0.50: return "moderate confidence"
    return "low confidence"

def _value_dir(prev, curr):
    d = curr - prev
    if d >  0.05: return "rising",  "anticipating future reward"
    if d < -0.05: return "falling", "not expecting a payoff"
    return "stable", "neutral about the immediate outcome"

def _dreamer_lines(exploration_bonus, world_model_score):
    lines = []
    wm = ("stable"    if world_model_score >= 0.75 else
          "uncertain" if world_model_score >= 0.40 else "low")
    wm_meaning = {
        "stable":    "the outcome matched its world-model prediction",
        "uncertain": "the outcome was somewhat surprising",
        "low":       "the agent was caught off-guard — its model mis-predicted this",
    }[wm]
    lines.append(f"World-model confidence was {wm}: {wm_meaning}.")
    if exploration_bonus > 0.70:
        lines.append("Exploration bonus was elevated — "
                     "the agent was in relatively unfamiliar territory.")
    return lines

def _ppo_lines(entropy, advantage):
    behaviour = ("exploring options" if entropy > 0.60
                 else "exploiting a known strategy")
    lines = [f"Policy entropy was {'high' if entropy > 0.60 else 'low'}, "
             f"indicating the agent was {behaviour}."]
    if advantage > 0.10:
        lines.append("Advantage was positive — this action was "
                     "better than the agent's baseline expectation.")
    elif advantage < -0.10:
        lines.append("Advantage was negative — this action underperformed "
                     "the agent's own expectations.")
    return lines

def generate_explanation(step, action, action_prob,
                         value_curr, value_prev,
                         algorithm="dreamer",
                         exploration_bonus=0.0, world_model_score=0.5,
                         entropy=0.5, advantage=0.0,
                         achievement_unlocked=None):
    conf          = _confidence(action_prob)
    vdir, vmeaning = _value_dir(value_prev, value_curr)

    lines = [
        f"At step {step}, the agent chose to {action} with {conf} "
        f"(action probability: {action_prob:.2f}).",
        f"Its value estimate was {vdir}, meaning it was {vmeaning}.",
    ]

    if algorithm == "dreamer":
        lines += _dreamer_lines(exploration_bonus, world_model_score)
    elif algorithm == "ppo":
        lines += _ppo_lines(entropy, advantage)

    if achievement_unlocked:
        lines.append(
            f"✅ This step unlocked the '{achievement_unlocked}' achievement.")

    return "  ".join(lines)
```

***

### Changes to `vis/data_manager.py`

Add **one function** at the bottom — algorithm auto-detection on CSV load:

```python
# Add to VisDataManager (or as a module-level function)

def detect_algorithm(df: pd.DataFrame) -> str:
    """Infer agent type from available CSV columns."""
    dreamer_cols = {"exploration_bonus", "world_model_score"}
    ppo_cols     = {"entropy", "advantage"}
    cols = set(df.columns)
    if dreamer_cols.issubset(cols):  return "dreamer"
    if ppo_cols.issubset(cols):      return "ppo"
    return "unknown"
```

Call it right after you read the CSV in your existing `load()` method:

```python
self.df = pd.read_csv(path)
self.algorithm = detect_algorithm(self.df)   # ← add this line
```

***

### Changes to `vis/widgets.py`

This file needs three targeted edits:

**1. Dynamic panel title** — find where `"Decision Attribution – Dreamer"` is set as a title string and replace:

```python
# Before
title = "Decision Attribution – Dreamer"

# After
algo_label = {"dreamer": "DreamerV2", "ppo": "PPO", "unknown": "Agent"}.get(
    self.algorithm, "Agent")
title = f"Decision Attribution – {algo_label}"
```

**2. Dynamic signal selection** — find where `exploration_bonus` and `world_model_score` are fetched for plotting and wrap in a conditional:

```python
if self.algorithm == "dreamer":
    signals = {
        "Value Estimate":   self.data["value_estimate"],
        "Action Prob.":     self.data["action_probability"],
        "Exploration Bonus":self.data["exploration_bonus"],
        "World-Model Score":self.data["world_model_score"],
    }
elif self.algorithm == "ppo":
    signals = {
        "Value Estimate":   self.data["value_estimate"],
        "Action Prob.":     self.data["action_probability"],
        "Entropy":          self.data["entropy"],
        "Advantage":        self.data["advantage"],
    }
else:
    # fallback: only plot what's present
    base = ["value_estimate", "action_probability"]
    signals = {col: self.data[col] for col in base if col in self.data.columns}
```

**3. Add the NLP explanation label** — directly below the Decision Attribution plot widget, add a `QLabel`:

```python
# In the Decision Attribution widget __init__
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

self.explanation_label = QLabel("")
self.explanation_label.setWordWrap(True)
self.explanation_label.setStyleSheet(
    "font-size: 11px; color: #333; padding: 6px; "
    "background: #f9f9f9; border-top: 1px solid #ddd;"
)
self.explanation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
self.explanation_label.setMinimumHeight(70)
self.layout().addWidget(self.explanation_label)   # add after the plot
```

Then in your existing step-update method, add:

```python
from vis.explainer import generate_explanation

row      = self.data.iloc[step_idx]
prev_row = self.data.iloc[max(0, step_idx - 1)]

text = generate_explanation(
    step              = step_idx,
    action            = row.get("executed_action", row.get("action", "unknown")),
    action_prob       = float(row.get("action_probability", 0.5)),
    value_curr        = float(row.get("value_estimate", 0.5)),
    value_prev        = float(prev_row.get("value_estimate", 0.5)),
    algorithm         = self.algorithm,
    exploration_bonus = float(row.get("exploration_bonus", 0.0)),
    world_model_score = float(row.get("world_model_score", 0.5)),
    entropy           = float(row.get("entropy", 0.5)),
    advantage         = float(row.get("advantage", 0.0)),
    achievement_unlocked = row.get("achievement", None),
)
self.explanation_label.setText(text)
```

***

### Changes to `vis/main.py`

Only one addition: after loading data, pass `algorithm` down to all widgets that need it:

```python
# After data load
algo = self.data_manager.algorithm
self.vis_widget.set_algorithm(algo)         # right panel plots
self.info_panel.set_algorithm(algo)         # optional — for label in achievements
```

Add a `set_algorithm(self, algo)` method to any widget that needs it — it just stores `self.algorithm = algo` and re-renders the title/signals.

***

### `SemanticEventDetector.py` — Wire It In

Your README notes this is currently **not wired into the active pipeline**. The XAI template is a natural trigger to activate it: fire `SemanticEventDetector` on load, and use its output to populate the `achievement_unlocked` field passed to `generate_explanation()`. This closes the loop between semantic events and the NLP explanation.

***

## Summary of Files to Touch

| File | Change |
|---|---|
| `vis/explainer.py` | **New file** — template engine |
| `vis/data_manager.py` | Add `detect_algorithm()`, store `self.algorithm` |
| `vis/widgets.py` | Dynamic title, dynamic signals, add `QLabel` |
| `vis/main.py` | Pass `algorithm` to widgets after load |
| `SemanticEventDetector.py` | Wire into load path to supply achievement events |