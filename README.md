# AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction

This repository contains the code and data for the paper **"AP-Based LLM Story Generation: Envisioning Technology Development through Science Fiction"**.

## Repository Contents

### Main Code Files

- **`full system/`** - Main story generation system implementation
- **`pick_topics.py`** - Script to select 10 topics from HANNA dataset for experiments
- **`evaluate.py`** - Evaluate generated stories using fine-tuned GPT-4o
- **`check_diversity.py`** - Calculate diversity metrics (Self-BLEU based)
- **`effect.py`** - Calculate parametric effect sizes (Cohen's d) for statistical analysis

### Data & Results

- **`stories/`** - Generated stories organized by method (baseline, ap_only, full_system)
- **`diversity_record/`** - Diversity analysis results for all experimental conditions
- **`gpt4o_eval_on_hanna_results.json`** - Base GPT-4o evaluation results on HANNA dataset
- **`finetuned_gpt4o_eval_on_hanna_results.json`** - Fine-tuned GPT-4o evaluation results
- **`statistical_test_result1.txt`** - Quality comparison statistical test results (Table 2)
- **`statistical_test_result2.txt`** - Diversity comparison statistical test results (Table 3)

## Usage

### Generate Stories
```bash
cd "full system"
python main.py --topic "Umbrella"
```

### Evaluate Stories
```bash
python evaluate.py --input_file stories/umbrella_story.json
```

### Calculate Diversity
```bash
python check_diversity.py --story_dir stories/
```

### Statistical Analysis
```bash
python effect.py  # Calculate effect sizes from diversity results
```

## License

MIT License