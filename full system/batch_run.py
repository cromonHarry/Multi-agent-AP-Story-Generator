import os
import traceback
import concurrent.futures
from openai import OpenAI
from config import OPENAI_API_KEY, NUM_AGENTS, NUM_ITERATIONS, MAX_CONCURRENT_STORIES
from ap_builder import APBuilder
from story_generator import StoryGenerator

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in config.py")

# The OpenAI client is thread-safe, so a single shared instance is fine here
global_client = OpenAI(api_key=OPENAI_API_KEY)

def process_single_story(theme, index, output_dir):
    try:
        print(f"  [Story {index} | START] Processing {theme}...")

        # Each story needs its own APBuilder and StoryGenerator instances
        # because AgentManager stores per-run agent state internally
        local_builder = APBuilder(global_client)
        local_gen = StoryGenerator(global_client)

        stage3_model = local_builder.generate_future_stage_multi_agent(tech_topic=theme)
        all_stages_data = {"Stage 3": stage3_model}

        final_outline = local_gen.generate_outline(all_stages_data)

        filename = f"{theme.replace(' ', '_')}_story_{index:02d}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(final_outline)

        print(f"  [Story {index} | DONE] Saved to {filename}")
        return True, index

    except Exception as e:
        print(f"  [Story {index} | ERROR] Failed: {e}")
        traceback.print_exc()
        return False, index

def run_batch_generation():
    themes = ["Grocery", "Password", "Soccer", "Smartphone"]
    stories_per_theme = 100

    print(f"=== Starting Batch Generation ===")
    print(f"Agents: {NUM_AGENTS} | Iterations: {NUM_ITERATIONS} | Max Concurrent: {MAX_CONCURRENT_STORIES}")
    print(f"Themes: {themes}")
    print("=" * 50)

    for theme in themes:
        print(f"\n>>> Processing Theme: {theme}")

        folder_name = f"{theme.replace(' ', '_')}_A{NUM_AGENTS}_I{NUM_ITERATIONS}"
        output_dir = os.path.join("batch_stories_ablation", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STORIES) as executor:
            future_to_index = {
                executor.submit(process_single_story, theme, i, output_dir): i
                for i in range(1, stories_per_theme + 1)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  [Story {index}] Exception: {exc}")

    print("\n" + "=" * 50)
    print("BATCH GENERATION COMPLETE")
    print("Check the 'batch_stories_ablation' folder.")
    print("=" * 50)

if __name__ == "__main__":
    run_batch_generation()
