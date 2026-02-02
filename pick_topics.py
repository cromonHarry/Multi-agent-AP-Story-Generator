import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ================= 配置区域 =================
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # your api
# ===========================================

client = OpenAI()

# 1. 准备 100 个常见的日常物品单词 (模拟输入)
words_pool = [
    "Toothbrush", "Coffee", "Subway", "Smartphone", "Umbrella", "Wallet", "Socks", "Chair", "Key", "Alarm", "Pan", "Receipt", "Towel", "Mirror", "Battery", "Soap", "Book", "Window", "Pillow", "Trash",
    "toothbrush","stovetop","wallet","smartphone","calendar","mask","backpack","umbrella","charger","laundry","coffee","key","notebook","refrigerator","bicycle","television","lamp","detergent","slippers","potted plant",
    "breakfast", "commute", "email", "shower", "budget", "exercise", "laundry", "grocery", "meeting", "coffee", "smartphone", "bedtime", "medication", "weather", "parking", "deadline", "dentist", "recycling", "password", "thermostat",
    "toothbrush","stove","wallet","calendar","smartphone","umbrella","chair","notebook","keys","coffee","laundry","thermostat","bus","lamp","medicine","recycling","password","grocery","alarm","exercise",
    "smartphone", "bread", "exercise", "car", "book", "money", "movie", "chair", "shirt", "rain", "email", "meeting", "soccer", "tree", "clock", "cart", "parent", "lock", "battery", "recycle",


]

def get_embeddings(text_list):
    """批量获取 Embedding 向量"""
    # OpenAI 限制单次请求大小，如果列表很大需要分批，这里100个可以直接发
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return np.array([data.embedding for data in response.data])

def select_most_diverse(words, embeddings, n_select=10):
    """
    使用贪心算法选择最多样化的 n 个词
    算法：Max-Min Similarity (寻找离当前集合最远的词)
    """
    n_total = len(words)
    # 计算所有词之间的相似度矩阵 (100x100)
    sim_matrix = cosine_similarity(embeddings)
    
    selected_indices = []
    
    # --- 第一步：找到初始差异最大的一对 ---
    # 将对角线（自己对自己）设为最大值，避免选中
    np.fill_diagonal(sim_matrix, 2.0)
    
    # 找到矩阵中相似度最小（值最小）的索引位置
    min_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    idx1, idx2 = min_idx
    
    selected_indices.extend([idx1, idx2])
    print(f"🚀 初始基准词 (差异最大的一对): [{words[idx1]}] & [{words[idx2]}] (相似度: {sim_matrix[idx1][idx2]:.4f})")
    
    # --- 第二步：循环补全剩余名额 ---
    while len(selected_indices) < n_select:
        remaining_indices = [i for i in range(n_total) if i not in selected_indices]
        
        # 对于每个候选词，我们要看它和“已选集合”有多像
        # 我们取它和已选集合中所有词的相似度的最大值（即它离集合最近的距离）
        # 然后我们要选出这个“最近距离”最远的那个候选词（Min-Max Strategy）
        
        candidates_max_similarity = []
        
        for i in remaining_indices:
            # 计算候选词 i 与所有已选词的相似度
            sims_to_selected = sim_matrix[i, selected_indices]
            # 找出它与现有集合最相似的程度
            max_sim = np.max(sims_to_selected)
            candidates_max_similarity.append(max_sim)
        
        # 在所有候选者的“最大相似度”中，找一个最小的
        # 意思是：这个词虽然和已选集合有关联，但它是所有候选词里“最不沾边”的
        best_candidate_idx_in_remaining = np.argmin(candidates_max_similarity)
        best_candidate_global_idx = remaining_indices[best_candidate_idx_in_remaining]
        
        selected_indices.append(best_candidate_global_idx)
        print(f"➕ 添加第 {len(selected_indices)} 个词: [{words[best_candidate_global_idx]}] (它与现有集合的最大相似度仅为: {candidates_max_similarity[best_candidate_idx_in_remaining]:.4f})")

    return [words[i] for i in selected_indices]

def main():
    print(f"📥 正在获取 {len(words_pool)} 个单词的 Embeddings...")
    embeddings = get_embeddings(words_pool)
    
    print("\n🔍 正在进行多样性筛选算法...")
    final_list = select_most_diverse(words_pool, embeddings, n_select=10)
    
    print("\n" + "="*40)
    print("✅ 最终选出的 10 个最具多样性的单词：")
    print("="*40)
    for i, word in enumerate(final_list, 1):
        print(f"{i}. {word}")

if __name__ == "__main__":
    main()