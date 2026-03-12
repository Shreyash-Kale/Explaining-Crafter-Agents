# Architecture Analysis: Issues & LLM-Powered Improvements

## Part 1: Major Architectural Issues

### Issue 1: **Temporal Misalignment (Hardcoded Workaround)**

**Current Approach:**
```python
# vis/data_manager.py
indices = np.linspace(0, num_training_samples - 1, episode_length).astype(int)
resampled_df = decision_df.iloc[indices].reset_index(drop=True)
```

**Problems:**
- ❌ Linear interpolation assumes smooth policy evolution (often false)
- ❌ Loses temporal resolution—decision made at step 150K might be assigned to wrong episode part
- ❌ No confidence bounds on resampled signals
- ❌ Bidirectional: both old training data can appear in new episodes AND new episodes miss coverage

**Impact:** Decisions show as averaged/smoothed rather than truthful snapshots

---

### Issue 2: **Sparse Decision Logging**

**Current Approach:**
```python
# dreamer/train.py, line 146
if i == 0 and step_count % 1000 < num_envs:  # Only every ~1000 training steps!
    attribution = policy.log_decision_attribution(obs, action_scalar)
```

**Problems:**
- ❌ ~200 decision samples for 250K training steps = 0.08% coverage
- ❌ Only logs when `i==0` (first environment only)
- ❌ Misses crucial decisions during exploration phase (early training)
- ❌ Can't analyze decision evolution within episodes

**Impact:** Decision attribution is statistically unreliable and incomplete

---

### Issue 3: **Episode-Level Ignorance (No Per-Episode Attribution)**

**Current System:**
```
Training Stream (Decision Attribution):
  Step 90K: [action, prob, value, explore]
  Step 91K: [action, prob, value, explore]
  ...

Episode Stream (Event Log):
  Step 0: action, reward, inventory
  Step 1: action, reward, inventory
  ...

Linking: ???  ← Requires post-hoc resampling
```

**Problems:**
- ❌ No true causal link between decisions and outcomes
- ❌ Can't say "this decision led to this reward"
- ❌ Makes ablation studies/counterfactuals impossible
- ❌ Violates episodic RL principle: decisions should be evaluated within episodes

**Impact:** Can't do true causal analysis

---

### Issue 4: **Limited Explainability (Numbers ≠ Explanations)**

**Current Attribution:**
- ✅ value_estimate: 0.142
- ✅ action_probability: 0.0581
- ✅ exploration_bonus: 2.833
- ❌ But WHY? What in the observation triggered these values?

There's no answer to:
- "Why did value drop at step 150?"
- "What visual features drove this decision?"
- "Is the agent exploring or just uncertain?"

---

### Issue 5: **Manual Semantic Event Detection**

```python
# SemanticEventDetector.py
# Hand-coded rules like:
if inventory['iron'] > 0 and time_step > 100:
    event = "COLLECTED_IRON"
```

**Problems:**
- ❌ Not scalable—requires manual coding for each event type
- ❌ Brittle—thresholds (time_step > 100) are hardcoded
- ❌ Misses cross-episode patterns
- ❌ Can't detect novel event types automatically

---

### Issue 6: **No Comparative Analysis**

**Current Limitation:**
- Can view one episode at a time
- No way to ask: "How do decision patterns differ between successful vs failed episodes?"
- No aggregation across episodes
- No pattern mining

---

### Issue 7: **Scalability Bottleneck (Single-Machine UI)**

```python
# vis/main.py: All data loaded into memory
self.event_df = pd.read_csv(csv_path)  # ← Full episode in memory
```

**Problems:**
- ❌ Can't visualize 10K episodes simultaneously
- ❌ No batch analysis
- ❌ Requires manual episode selection

---

## Part 2: LLM-Powered Improvements

### Improvement 1: **LLM-Generated Episode Narratives**

**Create human-readable summaries of episodes:**

```python
# New module: dreamer/episode_narrator.py

from openai import OpenAI

class EpisodeNarrator:
    def narrate_episode(self, csv_path, decision_df):
        """Convert episode data → natural language summary"""
        
        # Extract key events from CSV
        events = self.extract_events(csv_path)
        # e.g., [
        #   {"step": 5, "action": "collect_wood", "reward": 0.1},
        #   {"step": 42, "action": "place_table", "reward": 0.5},
        # ]
        
        # Get decision context
        decisions = self.extract_decisions(decision_df)
        # e.g., [
        #   {"step": 5, "value": 0.12, "explore": 2.83, "confidence": 0.06},
        # ]
        
        # Prompt LLM
        prompt = f"""
        Analyze this Crafter episode:
        
        Events: {events}
        Agent's Internal State: {decisions}
        
        Write a 3-sentence narrative explaining:
        1. What the agent tried to accomplish
        2. Key decisions and their outcomes
        3. Whether it explored or exploited
        
        Focus on cause-effect, not just listing events.
        """
        
        narrative = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return narrative.choices[0].message.content

# Usage in viz
episode_narrative = narrator.narrate_episode(csv_path, decision_df)
self.info_panel.set_narrative(episode_narrative)
```

**Result:**
```
Episode Summary:
"The agent explored wood-collecting early (high exploration_bonus), 
realized wood was valuable (value rose to 0.15), then exploited 
by repeatedly collecting wood. It successfully crafted a table at 
step 42, triggering the place_table achievement."
```

---

### Improvement 2: **Decision Explanation via LLM**

**When user hovers over a decision point, generate explain-ability:**

```python
class DecisionExplainer:
    def explain_decision(self, step, decision_data, obs_features):
        """
        decision_data = {
            "action": 9,  # do
            "action_prob": 0.058,
            "value": 0.12,
            "explore": 2.83,
            "world_model_score": 0.34
        }
        obs_features = extract_visual_features(frame)
        # e.g., ["player_near_tree", "inventory_full", "night_time"]
        """
        
        prompt = f"""
        The agent made this decision:
        - Action: DO (action_id=9)
        - Policy confidence: 5.8% (quite low)
        - Expected value: 0.12 (positive but not high)
        - Exploration bonus: 2.83 (high curiosity)
        - World-model confidence: 34% (uncertain about outcomes)
        
        Visual context: {obs_features}
        
        Explain in 1-2 sentences why this decision makes sense or doesn't.
        """
        
        explanation = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return explanation.choices[0].message.content

# Usage in UI
explanation = explainer.explain_decision(step, decision_data, obs_features)
label.setText(explanation)
```

**Result (on hover):**
```
"The agent chose DO with low confidence (5.8%) because it's exploring
(high curiosity: 2.83), not because it's confident the action will work.
The world-model is uncertain (34%), suggesting a novel situation."
```

---

### Improvement 3: **Automatic Semantic Event Detection**

**Replace hard-coded rules with LLM inference:**

```python
class SemanticEventDetector:
    def detect_events(self, csv_path):
        """
        Instead of:
            if inventory['iron'] > 0: event = "COLLECTED_IRON"
        
        Use LLM to identify patterns across episodes
        """
        
        # Load multiple episodes
        episodes = [pd.read_csv(f) for f in get_episode_csvs()]
        
        prompt = """
        Analyze these 10 game episodes (Crafter). For each, identify:
        1. Major milestones (e.g., "collected first diamond")
        2. Behavioral patterns (e.g., "explored before exploiting")
        3. Failure modes (e.g., "starved at step XYZ")
        4. Strategic objectives (e.g., "building was primary goal")
        
        Return JSON with discovered event types and their patterns.
        """
        
        events = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(events.choices[0].message.content)

# Output:
#{
#  "events": [
#    {"name": "first_resource_collected", "trigger": "any inventory > 0", "frequency": 0.95},
#    {"name": "crafting_attempt", "trigger": "place_table achievement", "frequency": 0.6},
#    {"name": "exploration_phase", "trigger": "high exploration_bonus + low value", "frequency": 0.8}
#  ]
#}
```

**Benefits:**
- ✅ Discovers event types automatically
- ✅ Scales to new environments without recoding
- ✅ Identifies behavioral patterns, not just threshold crossings

---

### Improvement 4: **Comparative Episode Analysis**

**Ask questions about episode relationships:**

```python
class ComparativeAnalyzer:
    def compare_episodes(self, episode_ids, question):
        """
        question: "Why did episodes 5 and 12 have such different rewards?"
        """
        
        episode_data = {}
        for ep_id in episode_ids:
            csv = load_episode(ep_id)
            episode_data[ep_id] = {
                "summary": extract_summary(csv),
                "decisions": extract_decision_patterns(csv),
                "achievements": get_achievements(csv),
                "reward": csv['cumulative_reward'].iloc[-1]
            }
        
        prompt = f"""
        Compare these episodes in Crafter:
        
        Episode 5 (reward: {episode_data[5]['reward']:.2f}):
        {episode_data[5]['summary']}
        
        Episode 12 (reward: {episode_data[12]['reward']:.2f}):
        {episode_data[12]['summary']}
        
        Question: {question}
        
        Analyze the decision patterns, achievements, and policies 
        to explain the difference.
        """
        
        analysis = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return analysis.choices[0].message.content

# Usage:
comparison = analyzer.compare_episodes(
    [5, 12],
    "Why did episode 5 collect more iron?"
)
print(comparison)
```

**Output:**
```
Episode 5 had 2x more iron because:
1. It explored longer early (exploration_bonus stayed high till step 80)
2. Found the stone quarry location earlier (placed stone at step 45)
3. Built a furnace by step 100, unlocking iron crafting
Episode 12 focused on wood-gathering, delaying furnace construction.
```

---

### Improvement 5: **Per-Episode Decision Attribution Logging (Engineering Fix)**

**Instead of resampling, log decisions per-episode:**

```python
# dreamer/train.py (IMPROVED)

def train_step_with_episode_logging(...):
    """Log decision attribution when episode ENDS, not based on training steps"""
    
    for i, env in enumerate(envs):
        obs, action, reward, done, info = env.step(action)
        
        # NEW: When episode terminates, log final state + decisions
        if done:
            # Compute decision attribution for the terminal state
            final_attribution = policy.log_decision_attribution(obs, action)
            
            # Link to episode ID (from info dict or filename)
            episode_id = info.get('episode_id')
            
            # Append to episode-specific decision log
            episode_decision_file = f"logs/decision_log_{episode_id}.csv"
            with open(episode_decision_file, 'a') as f:
                f.write(f"{episode_id},{final_attribution['action_taken']},"
                       f"{final_attribution['action_probability']},...\n")
            
            # Reset environment
            obs = env.reset()
```

**Benefits:**
- ✅ Direct episode-to-decision linking (no resampling)
- ✅ Solves temporal mismatch fundamentally
- ✅ Can log dense per-step decision data without overhead
- ✅ Enables true causal analysis

---

### Improvement 6: **Interactive Episode Querying**

**Let users ask questions in natural language:**

```python
class InteractiveAnalyzer:
    def query_episodes(self, user_query):
        """
        user_query: "Which episodes explored the most?"
        or: "Show me episodes where the agent regretted its decisions"
        or: "What's the decision pattern before achieving diamonds?"
        """
        
        # Search episodes using embeddings
        query_embedding = embed(user_query)
        
        episode_summaries = [
            {"id": i, "text": summarize_episode(i), "data": {...}}
            for i in range(500)
        ]
        
        # Find relevant episodes
        relevant = semantic_search(query_embedding, episode_summaries)
        
        # Ask LLM to aggregate findings
        prompt = f"""
        User asked: "{user_query}"
        
        Found {len(relevant)} relevant episodes:
        {relevant}
        
        Synthesize findings. What's the answer to their question?
        """
        
        answer = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return answer.choices[0].message.content

# Usage in UI
analyzer = InteractiveAnalyzer()
result = analyzer.query_episodes("When does the agent take risks?")
# Output: "The agent takes risks in episodes 5, 8, 12... [pattern analysis]"
```

---

### Improvement 7: **Real-Time Training Narrative Generation**

**As training progresses, stream LLM-generated insights:**

```python
# dreamer/train.py (LIVE MONITORING)

def training_loop(...):
    for step in range(total_steps):
        # ... normal training ...
        
        # Every 50K steps, generate narrative of what agent learned
        if step % 50000 == 0 and step > 0:
            recent_episodes = load_last_n_episodes(10)
            metrics = compute_metrics(recent_episodes)
            
            prompt = f"""
            After {step} training steps on Crafter, the agent shows:
            - Average reward: {metrics['avg_reward']:.2f}
            - Achievements learned: {metrics['achievements']}
            - Exploration vs exploitation: {metrics['explore_ratio']:.1%}
            - Value estimates: {metrics['avg_value']:.2f}
            
            What has the agent learned? What are its current capabilities?
            """
            
            insight = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log to training dashboard
            log_insight(step, insight.choices[0].message.content)
            print(f"[Step {step}] {insight.choices[0].message.content}")
```

**Output (streamed to dashboard):**
```
[Step 50K] The agent has learned basic resource gathering (wood, 
stone) and understands the table-crafting chain. It's starting to 
explore the furnace recipe but hasn't achieved iron crafting yet.

[Step 100K] Iron crafting is now reliable (achievement happens in 
80% of episodes). The agent is exploring higher-tier weapons but 
not yet attempting diamond-tier tasks.

[Step 200K] Advanced strategies emerging: selective exploration of 
rare resources, planning multi-step achievement chains. However, 
still relies heavily on exploration bonus (low exploitation).
```

---

## Part 3: Priority Ranking

| Issue | Severity | LLM Required | Effort | Impact |
|-------|----------|--------------|--------|--------|
| Temporal Misalignment | 🔴 Critical | ❌ (engineering fix) | Medium | High |
| Sparse Decision Logging | 🔴 Critical | ❌ (engineering fix) | Low | High |
| Per-Episode Attribution Loss | 🟠 High | ❌ (engineering fix) | Low | High |
| No Explainability | 🟡 Medium | ✅ (LLM core) | Low | Medium |
| Semantic Event Detection | 🟡 Medium | ✅ (LLM core) | Medium | Medium |
| Comparative Analysis | 🟡 Medium | ✅ (LLM core) | Medium | Medium |
| Single-Machine Bottleneck | 🟢 Low | ❌ (infra) | High | Low |

---

## Recommended Implementation Order

1. **First**: Fix per-episode decision attribution logging (engineering, fixes temporal issues)
2. **Second**: Add per-step logging (not just terminal)
3. **Third**: Integrate LLM-generated narratives
4. **Fourth**: Add decision explanations (on-hover)
5. **Fifth**: Implement semantic event detection
6. **Sixth**: Build comparative analyzer (requires batch infrastructure)

---

## Summary: How LLMs Add Value

| Feature | Without LLM | With LLM |
|---------|------------|----------|
| Understanding episodes | "value→0.12, explore→2.83, ..." | "Agent explored cautiously, found the task harder than expected" |
| Finding patterns | Manual code + thresholds | Automatic discovery across 100s episodes |
| Comparing episodes | Read each individually | "These 3 episodes failed because..." |
| Debugging training | Stare at reward curves | "Step 150K: agent shifted from exploration to exploitation" |
| Actionability | Raw numbers | Interpretable English causes |

**Key Insight**: LLMs don't replace metrics—they translate metrics into human insight.

---
