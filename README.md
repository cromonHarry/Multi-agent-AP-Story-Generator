# AP-Based LLM Story Generation

Repository for the paper: **"AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction"**

This project implements a multi-agent pipeline that generates science fiction stories grounded in the **Archaeological Prototyping (AP)** sociocultural model. Given a technology theme (e.g., "Smartphone"), the system builds a structured model of how that technology transforms society in the future, then uses that model as the backbone for a coherent sci-fi narrative.

**This paper is currently under review for ICEC 2026.**

---

## How It Works

The generation pipeline has three stages:

**1. Multi-Agent AP Model Building**
Multiple expert agents with different perspectives brainstorm each of the 18 AP model elements (6 societal objects + 12 transformation arrows). In each iteration, agents propose ideas in parallel, a judge selects the best one, and after all iterations a final judge picks the overall winner. This produces a structured JSON model of the technology's future society.

**2. Story Setting & Characters**
A Setting Agent designs the world and 4 key characters based on the AP model. A Global Overseer reviews the result against the AP model and sends feedback if inconsistencies are found (up to 3 retries).

**3. Story Outline Generation**
An Outline Agent writes the five narrative beats (Exposition → Rising Action → Climax → Falling Action → Resolution) sequentially, with the same Overseer review loop after each beat. The final output is a plain-text five-paragraph story.

---

## Setup

**Install dependencies:**
```bash
pip install openai tavily-python scipy scikit-learn numpy
```

**Set your API key** in `full system/config.py`:
```python
OPENAI_API_KEY = "your key here"
```

**Key parameters** (also in `config.py`):

| Parameter | Default | Description |
|---|---|---|
| `NUM_AGENTS` | 3 | Number of expert agents per AP element |
| `NUM_ITERATIONS` | 3 | Brainstorming rounds per element |
| `MAX_CONCURRENT_STORIES` | 5 | Parallel threads for batch generation |

---

## Usage

### Generate a single story
```bash
cd "full system"
python main.py
```
You will be prompted to enter a technology theme. The AP model is saved as `ap_model_<theme>.json` and the story as `story_outline_<theme>.txt`.

### Batch generation
Generates 100 stories for each of 4 themes (Grocery, Password, Soccer, Smartphone) in parallel:
```bash
cd "full system"
python batch_run.py
```
Output is saved to `batch_stories_ablation/<theme>_A<agents>_I<iterations>/`.

---

## Evaluation & Analysis

These scripts are run from the **repository root**, not from `full system/`.

**Story quality evaluation** — scores each story on 6 criteria (Relevance, Coherence, Empathy, Surprise, Engagement, Complexity) using a fine-tuned GPT-4o evaluator, with 20 concurrent threads:
```bash
python evaluate.py
# Output: evaluation_final.csv
```

**Diversity analysis** — computes diversity as `1 - mean(cosine similarity)` over story embeddings stored in `diversity_record/`, then runs Wilcoxon signed-rank tests:
```bash
python check_diversity.py
```

**Effect size analysis** — runs non-parametric effect size measures (rank-biserial correlation, Cliff's delta, Vargha-Delaney A) comparing baseline vs. AP model approach:
```bash
python effect.py
# Output: nonparametric_effect_size_results.json
```

**Topic selection** — selects the most semantically diverse topics from a word pool using a greedy Max-Min cosine similarity algorithm:
```bash
python pick_topics.py
```

---

## Project Structure

```
├── full system/          # Core generation pipeline
│   ├── main.py           # Entry point for single story generation
│   ├── batch_run.py      # Parallel batch generation
│   ├── ap_builder.py     # Orchestrates AP model construction
│   ├── agent_manager.py  # Multi-agent brainstorming logic
│   ├── story_generator.py# Setting + outline generation with Overseer review
│   ├── search_service.py # Optional Tavily web search (unused in default mode)
│   ├── config.py         # API key, agent count, iteration count
│   └── utils.py          # JSON parsing helper
├── evaluate.py           # Story quality scoring
├── check_diversity.py    # Diversity metric calculation
├── effect.py             # Non-parametric effect size analysis
├── pick_topics.py        # Diverse topic selection
├── diversity_record/     # Embedding .npy files and diversity summaries
└── stories/              # Generated story outputs
```

---

## AP Model

The AP (Archaeological Prototyping) model describes how a technology reshapes society through 18 elements:

- **6 Objects**: Avant-garde Social Issues, People's Values, Social Issues, Technology and Resources, Daily Spaces and User Experience, Institutions
- **12 Arrows**: directional relationships between those objects (Media, Community Formation, Standardization, Communication, Organization, etc.)

The full definitions are in `full system/config.py` under `SYSTEM_PROMPT` and `AP_MODEL_STRUCTURE`.

---

## Results

Pre-computed results are included in the repository:

| File | Contents |
|---|---|
| `gpt4o_eval_on_hanna_results.json` | Baseline story quality evaluation scores |
| `finetuned_gpt4o_eval_on_hanna_results.json` | Fine-tuned evaluator scores |
| `nonparametric_effect_size_results.json` | Effect size comparison results |
| `statistical_test_result1.txt` | Quality comparison (Table 2 in paper) |
| `statistical_test_result2.txt` | Diversity comparison (Table 3 in paper) |
