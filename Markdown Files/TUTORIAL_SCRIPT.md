# Crafter AI Analysis Tool - Tutorial Script for Participants

## Part 1: Welcome & Context (2 minutes)

Welcome! Thank you for participating in this study. Today, you're going to become an **AI analyst**—someone who investigates and understands how artificial intelligence agents make decisions.

Specifically, you'll be analyzing AI agents playing **Crafter**, a real-time strategy game. Your job is **not** to play the game yourself, but to watch the agents play and use our visualization tool to understand *why* they make the decisions they do.

**Think of yourself as a detective.** The agent is the subject, the game is the crime scene, and you have tools to investigate its reasoning.

---

## Part 2: Quick Intro to Crafter (2 minutes)

### What is Crafter?

Crafter is a 2D top-down survival game where an AI agent must:
- **Survive**: Maintain health, hunger, and energy
- **Gather resources**: Collect wood, stone, and other materials by breaking blocks
- **Craft items**: Combine materials to create better tools and equipment
- **Achieve goals**: Complete objectives like mining diamonds, defeating monsters, or building structures

### Why should you care?

The agent's decisions reveal its **priorities, strategy, and understanding of the game**. By analyzing these decisions, we can understand:
- Is the agent smart and strategic?
- Is it blindly following a random policy or making intentional choices?
- Does it prioritize long-term goals or short-term survival?
- Are its decisions explainable and predictable?

---

## Part 3: The Visualization Interface Tour (3-4 minutes)

### Interface Layout

The tool has three main components:

#### **Left Side: Video Player**
This shows the actual gameplay of an AI agent playing Crafter. 
- Use **Play/Pause** to control the video
- Use **Speed controls** to slow down or speed up the video (helpful for fast sequences)
- **Replay frequently!** You can watch the same segment multiple times to catch details

**What to look for:**
- The agent's position (white/colored square)
- Its interactions with the environment (breaking blocks, picking up items, moving)
- What resources it's gathering at different times
- Any patterns in movement or behavior

#### **Right Side: Visualization Panels (Toggle between two views)**

**View 1: Plots & Decision Tracking**
This shows **multiple graphs over time** that help you understand the agent's internal state:

1. **Cumulative Reward/Score Graph** (top)
   - Shows the agent's overall progress over time
   - Upward trends = agent is improving or achieving goals
   - Flat sections = agent might be stuck or idle
   - Sharp increases = significant achievements or breakthroughs

2. **Resource Tracking Graphs** (middle section)
   - Health (red): How much health the agent has left
   - Food/Hunger (green): Food status (staying fed keeps health up)
   - Energy/Fatigue (yellow): How tired the agent is
   - Inventory items: Wood, stone, diamonds, etc.
   - **Why this matters**: These show what the agent is prioritizing. If health is dropping, it's not feeding itself properly.

3. **Significant Events Marker** (green highlights)
   - Green vertical lines mark important events or achievements
   - These are moments where something significant happened (found a resource, completed an objective, etc.)

**View 2: Agent Achievements Panel**
- Shows a **list of achievements** the agent has completed
- Divided into two tabs: "Completed" and "Available"
- Helps you understand what the agent has accomplished and what it could still do
- **Useful for ranking**: More achievements = potentially higher quality agent

#### **Bottom: Timeline Controller**
- Shows your **current position in the video** (where you are in time)
- The sync between video, graphs, and achievements means they all move together
- Use this to jump to specific moments if needed

---

## Part 4: Think-Aloud Method (2-3 minutes)

We want to understand **how you think** as you analyze the agent. We'll ask you to use the **think-aloud method**.

### What does this mean?

**Simply narrate your thoughts as you work.** It might feel awkward at first, but it's very helpful for us.

### Examples of what to say:

**✓ Good examples:**
- "I notice the agent's health is dropping... it seems to have stopped gathering food"
- "The cumulative reward jumped significantly here—something big must have happened"
- "The agent just broke a lot of blocks in a row... maybe it's looking for a specific resource"
- "I'm confused why it's staying in one spot for so long, let me rewatch that section"
- "It seems like the agent prioritizes gathering wood early on, then switches to mining"

**✓ What you might question:**
- "Why did it do that instead of something else?"
- "Does that decision make sense given the situation?"
- "What triggered that behavior change?"

**✓ Feel free to:**
- Pause the video to collect your thoughts
- Replay sections you find interesting or confusing
- Make notes on the paper provided
- Take your time—there's no rush

### What NOT to do:
- Don't stay silent for long periods (we need to know what you're thinking)
- Don't worry about being "right" or "making sense"—your honest thoughts are what matter
- Don't hold back confusion or doubts—those are valuable!

---

## Part 5: How to Complete an Analysis Task (4-5 minutes)

Each episode follows the same structure. Here's what you'll do:

### Step 1: Pre-Episode Questions (1-2 mins)
- I'll ask you a few questions about the scenario or your expectations
- These help us understand your hypotheses before you see the data
- Examples: "What do you think the agent will prioritize?" or "Do you expect this agent to be successful?"

### Step 2: Watch & Analyze (up to 10 minutes total)
- Watch the gameplay video while exploring the visualization
- **Don't feel pressured to watch linearly!** You can:
  - Pause and explore the graphs in detail
  - Replay interesting sections
  - Toggle between Info Panel and Plots view to compare
  - Use the timeline to jump to specific events
- **Time limit**: You have a maximum of 10 minutes per episode (I'll warn you when you're running low)
- **Your goal**: Form an understanding of the agent's behavior, strategy, and decision quality

**What you should try to answer:**
- What is the agent trying to do?
- Is it succeeding or failing?
- What are its priorities (survival vs. achievement vs. resource gathering)?
- Does its behavior change over time? Why?
- Are the decisions explainable?
- How would you rate this agent compared to others?

### Step 3: Use the Blank Sheet for Notes
- We've provided paper for note-taking
- Write down key observations, questions, or insights
- These will help you remember details when comparing agents later

### Step 4: Post-Episode Questionnaire (2-3 mins)
- You'll answer questions about your experience:
  - How clear was the agent's strategy?
  - How well did the visualization help you understand it?
  - How would you rate this agent's quality?
  - Any observations or insights?

---

## Part 6: Ranking & Comparison Tips

Later, you'll rank all agents you've analyzed. Here are factors to consider:

**Quality Indicators (Higher = Better):**
- More achievements completed
- Sustained cumulative reward growth
- Stable resource management (not starving, maintaining health)
- Clear, understandable strategy
- Adapts behavior when needed
- Efficient use of time

**Red Flags (Lower Quality):**
- Frequently dying or barely surviving
- No discernible strategy or very random behavior
- Getting stuck repeatedly in one location
- Wasting resources or time
- Incomprehensible decision patterns

**Remember:** You're assessing the **AI agent's decision-making quality and explainability**, not playing the game yourself.

---

## Part 7: Practice Time (5 minutes)

Now we're going to do a **quick practice run** together.

### What we'll do:
1. Load a sample episode
2. You'll watch and analyze for 2-3 minutes
3. I'll observe and answer any questions you have about the interface
4. This is a safe space to fumble around and get comfortable

### Focus on:
- Finding the Play/Pause button and timeline
- Toggling between the two visualization views
- Getting a feel for what information each graph provides
- Speaking your thoughts aloud (even if it feels strange!)

**Go ahead and start exploring. Ask me anything if something is confusing.**

---

## Important Reminders

✓ **You're the expert analyst**—trust your instincts and observations
✓ **There are no "wrong" interpretations** about what you see, as long as you can explain your reasoning
✓ **Take your time**—careful analysis is more valuable than rushing
✓ **Keep thinking aloud**—even small observations help us
✓ **Ask questions**—I'm here to help clarify the interface or game mechanics if needed
✓ **Feel free to take notes**—we provide paper specifically for this
✓ **Replay and reanalyze**—if you want to double-check something, go ahead!

---

## Questions Before We Begin?

Do you have any questions about:
- The interface and how to use it?
- What Crafter is and how the game works?
- What think-aloud method means?
- What you're supposed to be analyzing?
- Anything else?

If not, we'll start with the practice episode now. Ready? Let's dive in!
