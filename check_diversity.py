import numpy as np
import os
import re
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import json

# ================= Configuration =================
# Directories for two groups
BASELINE_DIR = "diversity_baseline"
AP_MODEL_DIR = "diversity_ap_model_only"

# Significance level
ALPHA = 0.05
# =================================================


def extract_theme_name(folder_name):
    """Extract pure theme name from folder name"""
    cleaned = re.sub(r'_Baseline$', '', folder_name)
    cleaned = re.sub(r'_A\d+_I\d+$', '', cleaned)
    return cleaned


def load_embeddings_from_folder(folder_path, embedding_type):
    """Load all embeddings of specified type from folder"""
    embeddings = []
    suffix = f"_{embedding_type}.npy"
    
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(suffix):
            emb_path = os.path.join(folder_path, f)
            emb = np.load(emb_path)
            embeddings.append(emb)
    
    return embeddings


def calculate_diversity_score(embeddings):
    """Calculate diversity score (1 - mean cosine similarity)"""
    if len(embeddings) < 2:
        return None
    
    embedding_matrix = np.array(embeddings)
    matrix = cosine_similarity(embedding_matrix)
    n = len(matrix)
    
    upper_triangle_indices = np.triu_indices(n, k=1)
    similarities = matrix[upper_triangle_indices]
    
    if len(similarities) == 0:
        return None
    
    return float(1 - np.mean(similarities))


def scan_and_calculate_diversity(root_dir):
    """Scan directory and calculate diversity for each topic"""
    results = {}
    
    if not os.path.exists(root_dir):
        print(f"[ERROR] Directory not found: {root_dir}")
        return {}
    
    print(f"[INFO] Scanning: {root_dir}")
    
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        npy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
        if not npy_files:
            continue
        
        keyword_embs = load_embeddings_from_folder(item_path, "keyword")
        summary_embs = load_embeddings_from_folder(item_path, "summary")
        
        if len(keyword_embs) < 2 or len(summary_embs) < 2:
            continue
        
        keyword_div = calculate_diversity_score(keyword_embs)
        summary_div = calculate_diversity_score(summary_embs)
        
        theme = extract_theme_name(item)
        results[item] = {
            "theme": theme,
            "keyword": keyword_div,
            "summary": summary_div,
            "count": len(keyword_embs)
        }
        
        print(f"  [OK] {item}: keyword={keyword_div:.4f}, summary={summary_div:.4f}")
    
    return results


def extract_paired_scores(baseline_results, method_results):
    """Extract paired scores matched by theme name"""
    baseline_theme_map = {extract_theme_name(k): k for k in baseline_results.keys()}
    method_theme_map = {extract_theme_name(k): k for k in method_results.keys()}
    
    common_themes = set(baseline_theme_map.keys()) & set(method_theme_map.keys())
    
    if not common_themes:
        print("[ERROR] No matching topics found!")
        return None
    
    print(f"\n[OK] Matched topics: {len(common_themes)}")
    
    data = {
        "topics": [],
        "baseline_keyword": [],
        "baseline_summary": [],
        "method_keyword": [],
        "method_summary": []
    }
    
    for theme in sorted(common_themes):
        baseline_folder = baseline_theme_map[theme]
        method_folder = method_theme_map[theme]
        
        data["topics"].append(theme)
        data["baseline_keyword"].append(baseline_results[baseline_folder]["keyword"])
        data["baseline_summary"].append(baseline_results[baseline_folder]["summary"])
        data["method_keyword"].append(method_results[method_folder]["keyword"])
        data["method_summary"].append(method_results[method_folder]["summary"])
    
    for key in data:
        if key != "topics":
            data[key] = np.array(data[key])
    
    return data


# ==================== Non-parametric Effect Size Calculations ====================

def rank_biserial_correlation(x, y):
    """
    Calculate rank-biserial correlation (r) for paired samples.
    This is the non-parametric effect size for Wilcoxon signed-rank test.
    
    Formula: r = 1 - (2W) / (n(n+1)/2)
    Where W is the Wilcoxon W statistic (sum of positive ranks or negative ranks, whichever is smaller)
    
    Alternative formula using W+ and W-:
    r = (W+ - W-) / (W+ + W-)
    
    Args:
        x: baseline scores
        y: treatment scores (AP_Model)
    
    Returns:
        r: rank-biserial correlation coefficient (-1 to 1)
    """
    diff = y - x
    
    # Remove zero differences
    diff_nonzero = diff[diff != 0]
    n = len(diff_nonzero)
    
    if n == 0:
        return 0.0, 0, 0, 0
    
    # Get absolute differences and their ranks
    abs_diff = np.abs(diff_nonzero)
    ranks = stats.rankdata(abs_diff)
    
    # Calculate W+ (sum of ranks for positive differences) and W- (sum of ranks for negative differences)
    positive_ranks = ranks[diff_nonzero > 0]
    negative_ranks = ranks[diff_nonzero < 0]
    
    W_plus = np.sum(positive_ranks) if len(positive_ranks) > 0 else 0
    W_minus = np.sum(negative_ranks) if len(negative_ranks) > 0 else 0
    
    # Rank-biserial correlation
    # r = (W+ - W-) / (W+ + W-)
    total_ranks = W_plus + W_minus
    if total_ranks == 0:
        r = 0.0
    else:
        r = (W_plus - W_minus) / total_ranks
    
    return r, W_plus, W_minus, n


def rank_biserial_from_wilcoxon(W, n):
    """
    Calculate rank-biserial correlation from Wilcoxon W statistic.
    
    Formula: r = 1 - (2W) / (n(n+1)/2)
    
    Args:
        W: Wilcoxon W statistic (the smaller of W+ and W-)
        n: number of non-zero differences
    
    Returns:
        r: rank-biserial correlation
    """
    max_W = n * (n + 1) / 2
    r = 1 - (2 * W) / max_W
    return r


def cliffs_delta(x, y):
    """
    Calculate Cliff's delta for paired samples.
    
    Cliff's delta = (# of pairs where y > x - # of pairs where x > y) / total pairs
    
    This is equivalent to the probability that a random y > random x,
    minus the probability that a random x > random y.
    
    Range: -1 to 1
    """
    n = len(x)
    count_greater = 0
    count_less = 0
    
    for i in range(n):
        if y[i] > x[i]:
            count_greater += 1
        elif y[i] < x[i]:
            count_less += 1
    
    delta = (count_greater - count_less) / n
    return delta, count_greater, count_less


def vargha_delaney_a(x, y):
    """
    Calculate Vargha-Delaney A statistic.
    
    A = P(Y > X) + 0.5 * P(Y = X)
    
    This is the probability that a randomly selected y is greater than
    a randomly selected x, plus half the probability they are equal.
    
    Range: 0 to 1 (0.5 means no difference)
    """
    n = len(x)
    count_greater = 0
    count_equal = 0
    
    for i in range(n):
        if y[i] > x[i]:
            count_greater += 1
        elif y[i] == x[i]:
            count_equal += 1
    
    A = (count_greater + 0.5 * count_equal) / n
    return A


def interpret_rank_biserial(r):
    """
    Interpret rank-biserial correlation.
    Thresholds based on standard conventions:
    |r| < 0.1: negligible
    |r| 0.1-0.3: small
    |r| 0.3-0.5: medium
    |r| >= 0.5: large
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(d):
    """
    Interpret Cliff's delta.
    Thresholds from Romano et al. (2006):
    |d| < 0.147: negligible
    |d| 0.147-0.33: small
    |d| 0.33-0.474: medium
    |d| >= 0.474: large
    """
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    elif abs_d < 0.33:
        return "small"
    elif abs_d < 0.474:
        return "medium"
    else:
        return "large"


def interpret_vargha_delaney(A):
    """
    Interpret Vargha-Delaney A.
    Thresholds:
    A around 0.5: negligible
    A 0.56-0.64 or 0.36-0.44: small
    A 0.64-0.71 or 0.29-0.36: medium
    A > 0.71 or < 0.29: large
    """
    if 0.44 <= A <= 0.56:
        return "negligible"
    elif 0.36 <= A < 0.44 or 0.56 < A <= 0.64:
        return "small"
    elif 0.29 <= A < 0.36 or 0.64 < A <= 0.71:
        return "medium"
    else:
        return "large"


def run_nonparametric_effect_size_analysis(baseline_scores, method_scores, metric_name):
    """Run comprehensive non-parametric effect size analysis"""
    print(f"\n{'='*70}")
    print(f"Non-parametric Effect Size Analysis: {metric_name} Diversity")
    print(f"Baseline vs AP_Model")
    print('='*70)
    
    n = len(baseline_scores)
    diff = method_scores - baseline_scores
    
    # Descriptive Statistics
    print(f"\n[Descriptive Statistics]")
    print(f"  Baseline:  n={n}, Median={np.median(baseline_scores):.4f}, Mean={np.mean(baseline_scores):.4f}")
    print(f"  AP_Model:  n={n}, Median={np.median(method_scores):.4f}, Mean={np.mean(method_scores):.4f}")
    print(f"  Diff:      Median={np.median(diff):.4f}, Mean={np.mean(diff):.4f}")
    
    # Normality Test
    print(f"\n[Normality Test] (Shapiro-Wilk)")
    _, p_baseline = stats.shapiro(baseline_scores)
    _, p_method = stats.shapiro(method_scores)
    print(f"  Baseline:  p={p_baseline:.4f} {'[Normal]' if p_baseline > 0.05 else '[Non-normal]'}")
    print(f"  AP_Model:  p={p_method:.4f} {'[Normal]' if p_method > 0.05 else '[Non-normal]'}")
    
    # Wilcoxon Signed-Rank Test
    print(f"\n[Wilcoxon Signed-Rank Test]")
    try:
        W_stat, p_wilcoxon = stats.wilcoxon(method_scores, baseline_scores)
        print(f"  W-statistic: {W_stat:.4f}")
        print(f"  p-value:     {p_wilcoxon:.6f}")
        print(f"  Result:      {'*** Significant (p < 0.05)' if p_wilcoxon < ALPHA else 'Not significant'}")
    except Exception as e:
        print(f"  Error: {e}")
        W_stat, p_wilcoxon = None, None
    
    # Non-parametric Effect Sizes
    print(f"\n[Non-parametric Effect Size Measures]")
    print("-"*70)
    
    # 1. Rank-biserial correlation (r)
    r, W_plus, W_minus, n_nonzero = rank_biserial_correlation(baseline_scores, method_scores)
    r_interp = interpret_rank_biserial(r)
    print(f"  Rank-biserial correlation (r): {r:.4f}  [{r_interp}]")
    print(f"    W+ (sum of positive ranks): {W_plus:.1f}")
    print(f"    W- (sum of negative ranks): {W_minus:.1f}")
    print(f"    n (non-zero differences):   {n_nonzero}")
    print(f"    Interpretation: r = 1 means all AP_Model > Baseline")
    print(f"                    r = -1 means all Baseline > AP_Model")
    print(f"                    r = 0 means no difference")
    
    # 2. Cliff's delta
    delta, count_greater, count_less = cliffs_delta(baseline_scores, method_scores)
    delta_interp = interpret_cliffs_delta(delta)
    print(f"\n  Cliff's delta:                 {delta:.4f}  [{delta_interp}]")
    print(f"    # AP_Model > Baseline: {count_greater}")
    print(f"    # Baseline > AP_Model: {count_less}")
    print(f"    # Ties:                {n - count_greater - count_less}")
    
    # 3. Vargha-Delaney A
    A = vargha_delaney_a(baseline_scores, method_scores)
    A_interp = interpret_vargha_delaney(A)
    print(f"\n  Vargha-Delaney A:              {A:.4f}  [{A_interp}]")
    print(f"    Interpretation: A = {A:.2f} means {A*100:.0f}% probability that")
    print(f"                    a random AP_Model score > random Baseline score")
    
    print("-"*70)
    
    # Interpretation Guide
    print(f"\n[Interpretation Guide for Rank-biserial r]")
    print(f"  |r| < 0.1:   negligible")
    print(f"  |r| 0.1-0.3: small")
    print(f"  |r| 0.3-0.5: medium")
    print(f"  |r| >= 0.5:  large")
    
    print(f"\n[Interpretation Guide for Cliff's delta]")
    print(f"  |d| < 0.147:     negligible")
    print(f"  |d| 0.147-0.33:  small")
    print(f"  |d| 0.33-0.474:  medium")
    print(f"  |d| >= 0.474:    large")
    
    # Recommendation
    print(f"\n[Recommendation for Paper]")
    print(f"  Report: r = {r:.2f} ({r_interp} effect)")
    print(f"  Or: Cliff's delta = {delta:.2f} ({delta_interp} effect)")
    
    return {
        "metric": metric_name,
        "n": n,
        "baseline_median": float(np.median(baseline_scores)),
        "baseline_mean": float(np.mean(baseline_scores)),
        "method_median": float(np.median(method_scores)),
        "method_mean": float(np.mean(method_scores)),
        "normality_baseline_p": float(p_baseline),
        "normality_method_p": float(p_method),
        "wilcoxon_W": float(W_stat) if W_stat else None,
        "wilcoxon_p": float(p_wilcoxon) if p_wilcoxon else None,
        "rank_biserial_r": float(r),
        "rank_biserial_interpretation": r_interp,
        "W_plus": float(W_plus),
        "W_minus": float(W_minus),
        "cliffs_delta": float(delta),
        "cliffs_delta_interpretation": delta_interp,
        "vargha_delaney_A": float(A),
        "vargha_delaney_interpretation": A_interp,
        "count_method_greater": int(count_greater),
        "count_baseline_greater": int(count_less),
        "count_ties": int(n - count_greater - count_less)
    }


def print_comparison_table(data):
    """Print per-topic comparison"""
    print(f"\n{'='*70}")
    print("Per-Topic Comparison: Baseline vs AP_Model")
    print('='*70)
    print(f"{'Topic':<15} | {'BL Keyword':<11} | {'AP Keyword':<11} | {'BL Summary':<11} | {'AP Summary':<11}")
    print('-'*70)
    
    for i, topic in enumerate(data["topics"]):
        print(f"{topic:<15} | {data['baseline_keyword'][i]:.4f}      | {data['method_keyword'][i]:.4f}      | {data['baseline_summary'][i]:.4f}      | {data['method_summary'][i]:.4f}")
    
    print('-'*70)
    print(f"{'MEAN':<15} | {np.mean(data['baseline_keyword']):.4f}      | {np.mean(data['method_keyword']):.4f}      | {np.mean(data['baseline_summary']):.4f}      | {np.mean(data['method_summary']):.4f}")
    print(f"{'MEDIAN':<15} | {np.median(data['baseline_keyword']):.4f}      | {np.median(data['method_keyword']):.4f}      | {np.median(data['baseline_summary']):.4f}      | {np.median(data['method_summary']):.4f}")


def generate_latex_output(results_keyword, results_summary):
    """Generate LaTeX formatted output"""
    print(f"\n{'='*70}")
    print("LaTeX Output for Paper")
    print('='*70)
    
    rk = results_keyword
    rs = results_summary
    
    # Effect size table
    latex = r"""
\begin{table}[h]
\centering
\caption{Non-parametric effect size analysis: Baseline vs AP Model}
\begin{tabular}{lccccc}
\toprule
Metric & $Mdn_{BL}$ & $Mdn_{AP}$ & $r$ & Cliff's $\delta$ & $A$ \\
\midrule
"""
    
    latex += f"Keyword & {rk['baseline_median']:.3f} & {rk['method_median']:.3f} & {rk['rank_biserial_r']:.2f} & {rk['cliffs_delta']:.2f} & {rk['vargha_delaney_A']:.2f} \\\\\n"
    latex += f"Summary & {rs['baseline_median']:.3f} & {rs['method_median']:.3f} & {rs['rank_biserial_r']:.2f} & {rs['cliffs_delta']:.2f} & {rs['vargha_delaney_A']:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $Mdn$ = Median, $r$ = rank-biserial correlation, $\delta$ = Cliff's delta, $A$ = Vargha-Delaney A.
\item Effect size interpretation: """ + f"Keyword r = {rk['rank_biserial_r']:.2f} ({rk['rank_biserial_interpretation']}), Summary r = {rs['rank_biserial_r']:.2f} ({rs['rank_biserial_interpretation']})." + r"""
\end{tablenotes}
\end{table}
"""
    print(latex)
    
    # In-text reporting
    print("\n[In-text reporting format]:")
    print(f"  Keyword: Wilcoxon W = {rk['wilcoxon_W']:.1f}, p = {rk['wilcoxon_p']:.4f}, r = {rk['rank_biserial_r']:.2f}")
    print(f"  Summary: Wilcoxon W = {rs['wilcoxon_W']:.1f}, p = {rs['wilcoxon_p']:.4f}, r = {rs['rank_biserial_r']:.2f}")
    
    return latex


def main():
    print("="*70)
    print("Non-parametric Effect Size Calculator")
    print("Baseline vs AP_Model Only")
    print("Using: Rank-biserial correlation (r), Cliff's delta, Vargha-Delaney A")
    print("="*70)
    
    # 1. Load data
    print(f"\n[1/2] Processing Baseline...")
    baseline_results = scan_and_calculate_diversity(BASELINE_DIR)
    
    if not baseline_results:
        print("[ERROR] Baseline data is empty")
        return
    
    print(f"\n[2/2] Processing AP_Model...")
    method_results = scan_and_calculate_diversity(AP_MODEL_DIR)
    
    if not method_results:
        print("[ERROR] AP_Model data is empty")
        return
    
    # 2. Extract paired scores
    data = extract_paired_scores(baseline_results, method_results)
    
    if data is None:
        return
    
    # 3. Print comparison table
    print_comparison_table(data)
    
    # 4. Run non-parametric effect size analysis
    results_keyword = run_nonparametric_effect_size_analysis(
        data["baseline_keyword"],
        data["method_keyword"],
        "Keyword"
    )
    
    results_summary = run_nonparametric_effect_size_analysis(
        data["baseline_summary"],
        data["method_summary"],
        "Summary"
    )
    
    # 5. Generate LaTeX
    latex = generate_latex_output(results_keyword, results_summary)
    
    # 6. Save results
    output = {
        "config": {
            "baseline_dir": BASELINE_DIR,
            "method_dir": AP_MODEL_DIR
        },
        "keyword": results_keyword,
        "summary": results_summary
    }
    
    with open("nonparametric_effect_size_results.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[INFO] Results saved to nonparametric_effect_size_results.json")
    
    # 7. Final Summary
    print(f"\n{'='*70}")
    print("Summary")
    print('='*70)
    print(f"\nKeyword Diversity:")
    print(f"  Baseline Median: {results_keyword['baseline_median']:.4f}")
    print(f"  AP_Model Median: {results_keyword['method_median']:.4f}")
    print(f"  Wilcoxon: W = {results_keyword['wilcoxon_W']:.1f}, p = {results_keyword['wilcoxon_p']:.4f}")
    print(f"  Rank-biserial r = {results_keyword['rank_biserial_r']:.2f} ({results_keyword['rank_biserial_interpretation']})")
    print(f"  Cliff's delta = {results_keyword['cliffs_delta']:.2f} ({results_keyword['cliffs_delta_interpretation']})")
    print(f"  AP_Model > Baseline in {results_keyword['count_method_greater']}/{results_keyword['n']} topics")
    
    print(f"\nSummary Diversity:")
    print(f"  Baseline Median: {results_summary['baseline_median']:.4f}")
    print(f"  AP_Model Median: {results_summary['method_median']:.4f}")
    print(f"  Wilcoxon: W = {results_summary['wilcoxon_W']:.1f}, p = {results_summary['wilcoxon_p']:.4f}")
    print(f"  Rank-biserial r = {results_summary['rank_biserial_r']:.2f} ({results_summary['rank_biserial_interpretation']})")
    print(f"  Cliff's delta = {results_summary['cliffs_delta']:.2f} ({results_summary['cliffs_delta_interpretation']})")
    print(f"  AP_Model > Baseline in {results_summary['count_method_greater']}/{results_summary['n']} topics")


if __name__ == "__main__":
    main()