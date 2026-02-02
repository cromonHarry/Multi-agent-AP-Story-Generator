import os
import json
from openai import OpenAI
from config import OPENAI_API_KEY
from ap_builder import APBuilder
from story_generator import StoryGenerator

def main():
    # 0. Setup
    if not OPENAI_API_KEY:
        print("Error: Please set OPENAI_API_KEY in config.py or environment variables.")
        return
    
    # Init OpenAI Client directly (No SearchService)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 1. User Input
    tech_input = input("Enter the Technology/Theme for Future Sci-Fi (e.g., Dream Recording): ").strip()
    if not tech_input:
        print("Input cannot be empty.")
        return

    print(f"\nInitializing SF Generator for: {tech_input}")
    print("Mode: Pure Future Generation (No historical search)")
    
    # Initialize Services
    ap_builder = APBuilder(client)
    story_gen = StoryGenerator(client)

    # 2. Build AP Model (Only Stage 3)
    stage3_model = ap_builder.generate_future_stage_multi_agent(
        tech_topic=tech_input
    )

    # 3. Save AP Data to Dictionary
    all_stages_data = {
        "Stage 3": stage3_model
    }

    # Save JSON (Optional, good for debugging)
    filename_safe = tech_input.replace(' ', '_')
    with open(f"ap_model_{filename_safe}.json", "w", encoding='utf-8') as f:
        json.dump(all_stages_data, f, indent=2, ensure_ascii=False)
    print("\nAP Model (Future) data saved to JSON.")

    # 4. Generate Story Outline (Pure 5 Paragraphs)
    outline = story_gen.generate_outline(all_stages_data)

    # Output Result
    print("\n" + "="*50)
    print("GENERATED SF STORY OUTLINE")
    print("="*50)
    print(outline)
    print("="*50)

    # Save Story
    with open(f"story_outline_{filename_safe}.txt", "w", encoding='utf-8') as f:
        f.write(outline)
    print(f"Story outline saved to story_outline_{filename_safe}.txt")

if __name__ == "__main__":
    main()