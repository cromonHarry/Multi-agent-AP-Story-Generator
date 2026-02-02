import os
import re
import json
import time
import concurrent.futures
from openai import OpenAI

# 初始化 Client
    
client = OpenAI()

# 评估维度的定义
CRITERIA = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]

def parse_scores(response_text):
    """
    使用正则表达式从 GPT 输出中提取分数。
    """
    scores = {}
    for criterion in CRITERIA:
        # 匹配 pattern: Criterion: [number] 或 Criterion: number
        pattern = f"{criterion}:\\s*\\[?(\\d)\\]?"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                scores[criterion] = max(1, min(5, score))
            except ValueError:
                scores[criterion] = 0
        else:
            scores[criterion] = 0 
    return scores

def evaluate_single_story(filepath, theme):
    """
    读取单个故事文件，调用 GPT-4o 进行评分
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            story_content = f.read().strip()
            
        if not story_content or len(story_content) < 50: # 跳过空文件或过短的文件
            return None

        prompt = f"""
You are a story evaluator, you will evaluate a story which is written based on the given background. You need to score the story by the following criteria: Relevance, Coherence, Empathy, Surprise, Engagement, and Complexity.
You only need to output the result in the following format:
Relevance: [1-5]
Coherence: [1-5]
Empathy: [1-5]
Surprise: [1-5]
Engagement: [1-5]
Complexity: [1-5]

Writing Prompt:
You need to write a creative science fiction which is related to {theme}, you need to show how {theme} will be in the future.

Story Content:
{story_content}
"""
        response = client.chat.completions.create(
            model="finetune-gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict and objective literary critic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        scores = parse_scores(result_text)
        
        # 验证解析是否成功
        if sum(scores.values()) == 0:
            print(f"  [Warning] Failed to parse scores for {os.path.basename(filepath)}.")
            return None
            
        return scores

    except Exception as e:
        print(f"  [Error] Evaluating {os.path.basename(filepath)} failed: {e}")
        return None

def clean_theme_name(folder_name):
    """
    从文件夹名提取主题。
    例如: "Shoes_A3_I3" -> "Shoes"
    例如: "Kitchen_Knife_A3_I3" -> "Kitchen Knife"
    """
    # 移除 _A{数字}_I{数字} 的后缀
    cleaned = re.sub(r'_A\d+_I\d+$', '', folder_name)
    # 将下划线替换回空格
    cleaned = cleaned.replace('_', ' ')
    return cleaned

def run_evaluation():
    root_dir = "batch_stories_ablation" # 请确保这是你的根目录文件夹名
    
    if not os.path.exists(root_dir):
        print(f"Directory '{root_dir}' not found. Please check the folder name.")
        return

    # 1. 扫描所有 txt 文件
    all_files = []
    print(f"Scanning '{root_dir}' for story files...")
    
    # 遍历所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        # 获取当前文件夹名称（作为主题来源）
        folder_name = os.path.basename(subdir)
        
        # 如果当前是在根目录下，跳过文件扫描（只扫描子文件夹里的文件）
        if subdir == root_dir:
            continue

        current_theme = clean_theme_name(folder_name)
        
        for file in files:
            # 宽松匹配：只要是 txt 且不是系统隐藏文件
            if file.lower().endswith(".txt") and not file.startswith("."):
                all_files.append({
                    "path": os.path.join(subdir, file),
                    "theme": current_theme,
                    "filename": file
                })

    total_files = len(all_files)
    print(f"Found {total_files} stories to evaluate.")
    
    # 简单的调试：打印前3个找到的文件，看看是否正确
    if total_files > 0:
        print("Example files found:")
        for i in range(min(3, total_files)):
            f = all_files[i]
            print(f"  - [{f['theme']}] {f['filename']}")
    else:
        print("No files found! Please check if 'batch_stories_ablation' folder contains .txt files.")
        return

    # 2. 多线程评估
    results = []
    eval_concurrency = 20 # 提高并发以加快速度
    
    print(f"\nStarting evaluation with {eval_concurrency} threads...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=eval_concurrency) as executor:
        future_to_file = {
            executor.submit(evaluate_single_story, item["path"], item["theme"]): item 
            for item in all_files
        }
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_file):
            item = future_to_file[future]
            completed_count += 1
            if completed_count % 50 == 0: # 每50个打印一次进度
                print(f"  Progress: {completed_count}/{total_files}...")
                
            scores = future.result()
            if scores:
                results.append(scores)

    # 3. 统计结果
    if not results:
        print("No valid results obtained.")
        return

    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Total Stories Evaluated: {len(results)}")
    
    avg_scores = {c: 0.0 for c in CRITERIA}
    for res in results:
        for c in CRITERIA:
            avg_scores[c] += res.get(c, 0)
    
    for c in CRITERIA:
        avg_scores[c] = avg_scores[c] / len(results)

    total_avg = sum(avg_scores.values()) / len(CRITERIA)

    # 4. 输出统计
    print("\n[Average Scores by Dimension]")
    for c in CRITERIA:
        print(f"{c:<15}: {avg_scores[c]:.2f}")
    
    print("-" * 30)
    print(f"{'OVERALL SCORE':<15}: {total_avg:.2f}")
    print("="*50)
    
    # 保存结果到 CSV
    with open("evaluation_final.csv", "w", encoding="utf-8") as f:
        f.write("Dimension,Score\n")
        for c in CRITERIA:
            f.write(f"{c},{avg_scores[c]:.2f}\n")
        f.write(f"Overall,{total_avg:.2f}\n")

    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_evaluation()