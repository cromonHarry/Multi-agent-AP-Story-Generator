import os
import traceback
import concurrent.futures
from openai import OpenAI
from config import OPENAI_API_KEY, NUM_AGENTS, NUM_ITERATIONS, MAX_CONCURRENT_STORIES
from ap_builder import APBuilder
from story_generator import StoryGenerator

# 初始化全局 Client (Client本身是线程安全的)
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in config.py")
    
global_client = OpenAI(api_key=OPENAI_API_KEY)

def process_single_story(theme, index, output_dir):
    """
    单个故事生成的独立工作单元。
    必须在函数内部实例化 APBuilder 和 StoryGenerator，
    以确保每个线程拥有独立的状态（特别是 AgentManager 的 agents 列表）。
    """
    try:
        print(f"  [Story {index} | START] Processing {theme}...")
        
        # 1. 创建独立的实例 (Crucial for Thread Safety)
        # 每个故事都有自己的一组 Agent 和 状态
        local_builder = APBuilder(global_client)
        local_gen = StoryGenerator(global_client)
        
        # 2. 生成 Future AP Model
        stage3_model = local_builder.generate_future_stage_multi_agent(
            tech_topic=theme
        )
        all_stages_data = {"Stage 3": stage3_model}

        # 3. 生成纯文本大纲
        final_outline = local_gen.generate_outline(all_stages_data)

        # 4. 保存文件
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
    # 主题列表
    themes = ["Grocery", "Password", "Soccer", "Smartphone"]
    # 每个主题生成的故事数量
    stories_per_theme = 100
    
    print(f"=== Starting Batch Generation (Parallel Mode) ===")
    print(f"Agents: {NUM_AGENTS} | Iterations: {NUM_ITERATIONS}")
    print(f"Max Concurrent Stories: {MAX_CONCURRENT_STORIES}")
    print(f"Themes: {themes}")
    print("="*50)

    for theme in themes:
        print(f"\n>>> Processing Theme: {theme}")
        
        # 创建文件夹
        folder_name = f"{theme.replace(' ', '_')}_A{NUM_AGENTS}_I{NUM_ITERATIONS}"
        output_dir = os.path.join("batch_stories_ablation", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用 ThreadPoolExecutor 进行并发执行
        # 这将同时启动 MAX_CONCURRENT_STORIES 个线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STORIES) as executor:
            future_to_index = {}
            
            # 提交任务
            for i in range(1, stories_per_theme + 1):
                future = executor.submit(process_single_story, theme, i, output_dir)
                future_to_index[future] = i
            
            # 等待完成
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    success, _ = future.result()
                    # 可以在这里做进度统计
                except Exception as exc:
                    print(f"  [Story {index}] generated an exception: {exc}")

    print("\n" + "="*50)
    print("BATCH GENERATION COMPLETE")
    print("Check the 'batch_stories_ablation' folder.")
    print("="*50)

if __name__ == "__main__":
    run_batch_generation()