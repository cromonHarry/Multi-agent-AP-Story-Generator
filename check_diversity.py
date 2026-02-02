import os
import glob
import numpy as np
import concurrent.futures
import time
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ================= 配置区域 =================
# 故事根目录
DATA_ROOT = "batch_stories_ablation3_2"
# 并发线程数 (用于 GPT-4o 概念提取)
MAX_WORKERS = 20
# Embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"
# ===========================================


client = OpenAI()

def get_theme_from_folder(folder_name):
    """从文件夹名提取主题，例如 'Shoes_A3_I3' -> 'Shoes'"""
    cleaned = re.sub(r'_A\d+_I\d+$', '', folder_name)
    return cleaned.replace('_', ' ')

def read_all_stories(root_dir):
    """扫描所有子文件夹，按文件夹分组读取故事"""
    groups = {}
    
    if not os.path.exists(root_dir):
        print(f"❌ 错误: 找不到目录 {root_dir}")
        return {}

    print(f"📂 正在扫描 {root_dir} ...")
    
    for subdir, _, files in os.walk(root_dir):
        if subdir == root_dir:
            continue
            
        folder_name = os.path.basename(subdir)
        txt_files = [f for f in files if f.endswith(".txt") and not f.startswith(".")]
        
        if not txt_files:
            continue
            
        story_list = []
        for f in txt_files:
            path = os.path.join(subdir, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    story_list.append({"filename": f, "content": content, "path": path})
        
        if story_list:
            groups[folder_name] = story_list
            
    return groups

def extract_single_concept(story_data, theme):
    """
    提取单个故事的核心概念
    """
    try:
        # 如果内容太短，可能无效
        if len(story_data['content']) < 50:
            return None

        prompt = f"""
        You are a researcher analyzing sci-fi stories.
        The following is a story based on the theme: "{theme}".
        
        Task: Identify the specific futuristic evolution, function, or form of the "{theme}" described in this story.
        
        Constraint: Summarize the concept in NO MORE THAN 5 words. English Only.
        Focus on the functionality and social role. 
        Do NOT summarize the plot. Only summarize the object's setting.
        
        Story content:
        {story_data['content'][:2000]} 
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        concept = response.choices[0].message.content.strip()
        return concept
    except Exception as e:
        print(f"  [提取失败] {story_data['filename']}: {e}")
        return None

def process_group_concepts(folder_name, stories):
    """
    处理一个组（文件夹）内的所有故事：并发提取概念
    """
    theme = get_theme_from_folder(folder_name)
    print(f"\n🔍 处理分组: [{folder_name}] (主题: {theme}, 数量: {len(stories)})")
    
    concepts = []
    filenames = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_story = {
            executor.submit(extract_single_concept, story, theme): story 
            for story in stories
        }
        
        completed = 0
        total = len(stories)
        
        for future in concurrent.futures.as_completed(future_to_story):
            story = future_to_story[future]
            result = future.result()
            
            if result:
                concepts.append(result)
                filenames.append(story['filename'])
            
            completed += 1
            if completed % 20 == 0:
                print(f"  ...进度 {completed}/{total}")

    return concepts, filenames

def get_embeddings_batch(text_list, batch_size=50):
    """
    批量获取 Embeddings 以节省请求次数
    """
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            # 保证顺序一致
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"  [Embedding Error] Batch {i}: {e}")
            # 如果失败，填充零向量防止崩溃（或者重试）
            all_embeddings.extend([[0.0]*1536] * len(batch))
            
    return all_embeddings

def calculate_diversity_score(embeddings):
    """
    计算多样性分数 (1 - 平均余弦相似度)
    """
    if len(embeddings) < 2:
        return 0.0, np.array([])
        
    matrix = cosine_similarity(embeddings)
    n = len(matrix)
    
    # 取上三角（不含对角线）
    upper_triangle_indices = np.triu_indices(n, k=1)
    similarities = matrix[upper_triangle_indices]
    
    if len(similarities) == 0:
        return 0.0, matrix

    avg_similarity = np.mean(similarities)
    diversity_score = 1 - avg_similarity
    
    return diversity_score, matrix

def main():
    start_time = time.time()
    
    # 1. 读取所有故事
    groups = read_all_stories(DATA_ROOT)
    if not groups:
        print("未找到任何故事文件。")
        return

    report_data = []

    # 2. 逐组处理
    for folder_name, stories in groups.items():
        # A. 提取概念
        concepts, filenames = process_group_concepts(folder_name, stories)
        
        if len(concepts) < 2:
            print(f"  ⚠️ 跳过 {folder_name}: 有效概念少于 2 个")
            continue
            
        # B. 获取向量
        print(f"  🧬 计算 Embeddings ({len(concepts)} 个概念)...")
        embeddings = get_embeddings_batch(concepts)
        
        # C. 计算多样性
        diversity, matrix = calculate_diversity_score(embeddings)
        
        print(f"  ✅ 多样性得分: {diversity:.4f}")
        
        report_data.append({
            "folder": folder_name,
            "count": len(concepts),
            "diversity": diversity,
            "concepts": concepts,
            "filenames": filenames,
            "matrix": matrix
        })

    # 3. 输出最终报告
    print("\n" + "="*60)
    print("📊 概念多样性分析报告 (Concept Diversity Report)")
    print("="*60)
    print(f"{'Folder / Theme':<30} | {'Count':<6} | {'Diversity Score':<15}")
    print("-" * 60)
    
    total_diversity = 0
    valid_groups = 0
    
    for item in report_data:
        print(f"{item['folder']:<30} | {item['count']:<6} | {item['diversity']:.4f}")
        total_diversity += item['diversity']
        valid_groups += 1
        
    print("-" * 60)
    if valid_groups > 0:
        print(f"{'AVERAGE':<30} | {'-':<6} | {total_diversity/valid_groups:.4f}")
    
    # 4. 保存详细结果到 CSV
    with open("diversity_results.csv", "w", encoding="utf-8") as f:
        f.write("Folder,Filename,Extracted_Concept\n")
        for item in report_data:
            for fname, concept in zip(item['filenames'], item['concepts']):
                # 处理一下 concept 里的逗号或换行，防止 CSV 格式乱掉
                clean_concept = concept.replace('"', "'").replace('\n', ' ')
                f.write(f"{item['folder']},{fname},\"{clean_concept}\"\n")
                
    print(f"\n详细概念提取结果已保存至 diversity_results.csv")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()