# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a **Multi-Agent AP-Based Science Fiction Story Generator**. Given a technology theme, the system builds an Archaeological Prototyping (AP) sociocultural model and then generates a sci-fi story grounded in that model. This is associated with the paper: *"AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction"*.

## Running the System

**Single story generation:**
```bash
cd "full system"
python main.py
# Enter a technology theme when prompted (e.g., "Smartphone")
# Outputs: ap_model_[topic].json, story_outline_[topic].txt
```

**Batch generation (100 stories × 4 themes, parallel):**
```bash
cd "full system"
python batch_run.py
# Outputs to: batch_stories_ablation/[THEME]_A[agents]_I[iterations]/
```

**Analysis scripts (run from repo root):**
```bash
python evaluate.py        # Evaluate story quality with fine-tuned GPT-4o
python check_diversity.py # Compute diversity metrics from embeddings
python effect.py          # Compute Cohen's d effect sizes
python pick_topics.py     # Select maximally diverse topics from a word pool
```

## Configuration

All tunable parameters are in `full system/config.py`:
- `OPENAI_API_KEY` — set your key here (no .env support)
- `NUM_AGENTS` — number of expert agents per AP element (default: 3)
- `NUM_ITERATIONS` — brainstorming iterations per element (default: 3)
- `MAX_CONCURRENT_STORIES` — parallel threads for batch generation (default: 5)

## Dependencies

No requirements.txt exists. Install manually:
```bash
pip install openai tavily-python scipy scikit-learn numpy
```

## Architecture

The pipeline has three major stages:

### Stage 1 — AP Model Generation (`ap_builder.py`)
Builds the 18-element AP model (6 objects + 12 arrows) for the given technology theme. Each element is generated sequentially, with prior elements passed as context to subsequent ones. The result is a nested dict saved as `ap_model_[topic].json`.

### Stage 2 — Multi-Agent Brainstorming (`agent_manager.py`)
For each AP element:
1. `generate_agents()` uses `gpt-4o-mini` to create `NUM_AGENTS` diverse expert personas
2. Each agent brainstorms in parallel (ThreadPoolExecutor, temperature=1.2)
3. A judge selects the best proposal per round (temperature=0)
4. After `NUM_ITERATIONS` rounds, a final judge picks the best overall result

### Stage 3 — Story Generation (`story_generator.py`)
Given the completed AP model:
1. A Global Overseer extracts a setting brief and plot brief
2. A Setting Agent generates a world description + 4 characters (JSON), approved by the Overseer (up to 3 retries)
3. An Outline Agent writes each of 5 narrative beats (Exposition → Rising Action → Climax → Falling Action → Resolution), each approved by the Overseer (up to 3 retries per beat)
4. The 5 beats are compiled into a final plain-text story

### Supporting Components
- `utils.py` — `parse_json_response()` strips markdown fences from GPT JSON output
- `search_service.py` — Optional Tavily web search integration for historical AP data; unused in the default future-only generation mode
- `evaluate.py` — Evaluates stories using a fine-tuned GPT-4o model on 6 criteria (Relevance, Coherence, Empathy, Surprise, Engagement, Complexity); outputs `evaluation_final.csv`
- `check_diversity.py` / `effect.py` — Load `.npy` embedding files from `diversity_record/` and compute diversity as `1 - mean(cosine_similarity)`, plus statistical tests

## AP Model Structure

The AP (Archaeological Prototyping) model is a sociocultural framework. The 6 objects are: Avant-garde Social Issues, People's Values, Social Issues, Technology and Resources, Daily Spaces and User Experience, Institutions. The 12 arrows represent directional transformations between them (Media, Community Formation, Standardization, etc.), defined in `config.py` under `AP_MODEL_STRUCTURE`.

## Data Layout

- `stories/` — Generated story outputs organized by method/temperature variant
- `diversity_record/` — Per-method embedding vectors (`.npy`) and diversity summary JSONs; subdirs include `baseline`, `ap_model_only`, `full_system`, and ablation variants
- `batch_stories_ablation/` — Batch-run outputs, named `[THEME]_A[agents]_I[iterations]/`
