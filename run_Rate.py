# 整合的完整主流程代码，包括自动率定（遗传算法）+ 相似度计算 + MySQL 提取

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random,math
from collections import defaultdict
from util.ReadSave import save_to_csv
from mysql_info import read_config, execute_sql
from vector_storage import get_data_from_chroma
from mysql_info import ShResults, ShResultsP

# ------------------- 相似度计算逻辑 ------------------- #
def calculate_similarity(table, weights, whole_threshold, col_threshold_list, ids):
    try:
        valid_cols = [col for col, w in weights.items() if w > 0]
        if not valid_cols:
            print("没有有效字段用于计算相似度。")
            return None, None

        all_col_data = {}
        for col in valid_cols:
            all_col_data[col] = get_data_from_chroma(f"{table}_{col}", ids=ids)

        sample_col = next(iter(all_col_data))
        ids_list = all_col_data[sample_col]['ids']
        if not ids_list:
            print("ID 列表为空。")
            return None, None

        cols_similarity_filtered = {'ids': ids_list}
        high_similarity_pairs = defaultdict(list)

        for col, data in all_col_data.items():
            matrix = cosine_similarity(data["embeddings"])
            threshold = col_threshold_list.get(col, 1.0)
            filtered_matrix = np.where(matrix >= threshold, matrix, 0.0)
            cols_similarity_filtered[col] = filtered_matrix.tolist()

            for i in range(len(filtered_matrix)):
                for j in range(i + 1, len(filtered_matrix)):
                    sim = filtered_matrix[i, j]
                    if sim > 0:
                        high_similarity_pairs[col].append((ids_list[i], ids_list[j], round(sim, 2)))

        total_matrix = np.zeros((len(ids_list), len(ids_list)))
        for col, mat in cols_similarity_filtered.items():
            if col == 'ids': continue
            total_matrix += np.array(mat) * weights[col]

        weighted_similarities = []
        for i in range(len(total_matrix)):
            for j in range(i + 1, len(total_matrix)):
                score = total_matrix[i, j]
                if score >= whole_threshold:
                    weighted_similarities.append((ids_list[i], ids_list[j], round(score, 3)))

        matching_pairs = defaultdict(list)
        for col, pairs in high_similarity_pairs.items():
            pair_dict = {(id1, id2): sim for id1, id2, sim in pairs}
            for id1, id2, score in weighted_similarities:
                if (id1, id2) in pair_dict or (id2, id1) in pair_dict:
                    matching_pairs[col].append((id1, id2, pair_dict.get((id1, id2), pair_dict.get((id2, id1)))))

        return weighted_similarities, matching_pairs

    except Exception as e:
        print(f"[错误] 计算相似度失败: {e}")
        return None, None

# ------------------- MySQL 提取 ------------------- #
def fetch_similarity_details_from_mysql(table, whole_weighted_similarities, matching_pairs, db_config_path: str):
    all_ids = set()
    for id1, id2, _ in whole_weighted_similarities:
        all_ids.update([id1, id2])

    db_config = read_config(db_config_path)
    model = {'sh_results': ShResults, 'sh_results_p': ShResultsP}.get(table)

    df = execute_sql(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        model=model,
        ids=all_ids
    )

    if df.empty:
        return pd.DataFrame(), {}

    df.set_index('id', inplace=True)
    df_reason = pd.DataFrame(columns=['duplicate_id', 'similarity', 'id', 'scene_name', 'reason'])
    seen_pairs = {}

    for keyword, pairs in matching_pairs.items():
        for id1, id2, _ in pairs:
            key = frozenset((id1, id2))
            duplicate_id = f"{min(id1, id2)}_{max(id1, id2)}"
            similarity = next((sim for a, b, sim in whole_weighted_similarities if frozenset((a, b)) == key), None)

            if key in seen_pairs:
                idx = seen_pairs[key]
                df_reason.at[idx, 'reason'] += f", {keyword}"
                df_reason.at[idx + 1, 'reason'] += f", {keyword}"
            else:
                rows = [{
                    'duplicate_id': duplicate_id,
                    'similarity': similarity,
                    'id': id1,
                    'scene_name': df.at[int(id1), 'scene_name'],
                    'reason': keyword
                }, {
                    'duplicate_id': duplicate_id,
                    'similarity': similarity,
                    'id': id2,
                    'scene_name': df.at[int(id2), 'scene_name'],
                    'reason': keyword
                }]
                df_reason = pd.concat([df_reason, pd.DataFrame(rows)], ignore_index=True)
                seen_pairs[key] = len(df_reason) - 2

    dfs_content_pairs = defaultdict(list)
    for keyword, pairs in matching_pairs.items():
        for id1, id2, sim in pairs:
            for _id in [id1, id2]:
                dfs_content_pairs[keyword].append({
                    'duplicate_id': f"{min(id1, id2)}_{max(id1, id2)}",
                    'similarity': sim,
                    'id': _id,
                    'scene_name': df.at[int(_id), 'scene_name'],
                    'content': df.at[int(_id), keyword]
                })

    return df_reason, dfs_content_pairs

# ------------------- 遗传算法优化 ------------------- #
def fitness_function(weights, thresholds, whole_threshold, similarity_func, table, ids, ground_truth_pairs, negative_pairs=None, return_detail=False):
    try:
        sim_pairs, _ = similarity_func(table, weights, whole_threshold, thresholds, ids)
        if not sim_pairs:
            return (0.0, 0.0, 0.0) if return_detail else 0.0

        detected = set(frozenset((str(id1), str(id2))) for id1, id2, _ in sim_pairs)
        truth = set(frozenset((str(a), str(b))) for a, b in ground_truth_pairs)
        negatives = set(frozenset((str(a), str(b))) for a, b in (negative_pairs or []))

        # print("Sample detected:", list(detected)[:3])
        # print("Sample negatives:", list(negatives)[:3])

        tp = len(detected & truth)
        fp = len(detected & negatives)
        fn = len(truth - detected)
        # print("tp:",tp,"fp:",fp,"fn:",fn)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return (f1, precision, recall) if return_detail else f1
        # # 引入负样本惩罚项
        # # penalty = fp / (len(negatives) + 1e-6)
        # # score = f1 - 0.8 * penalty  # 惩罚系数可调（越大惩罚越重）

        # # 版本二：非线性惩罚（替换上一行）
        # penalty = fp
        # print("惩罚值：", penalty)
        # score = f1 - 0.08 * penalty
        # print("适应度：", score)

        # return (score, precision, recall) if return_detail else score
    except:
        return (0.0, 0.0, 0.0) if return_detail else 0.0


def random_config(fields, weight_ranges, threshold_ranges, whole_t_range):
    # 每个字段独立设定权重
    weights = {}
    for f in fields:
        low, high = weight_ranges.get(f, (0.01, 0.25))
        weights[f] = random.uniform(low, high)
    s = sum(weights.values())
    weights = {f: w / s for f, w in weights.items()}

    # 每个字段独立设定阈值
    thresholds = {}
    for f in fields:
        low, high = threshold_ranges.get(f, (0.5, 0.95))
        thresholds[f] = round(random.uniform(low, high), 2)

    # 全局阈值范围
    whole_threshold = round(random.uniform(*whole_t_range), 2)

    return weights, thresholds, whole_threshold

def genetic_optimize(
    table,
    ids,
    ground_truth_pairs,
    fields,
    similarity_func,
    pop_size,
    generations,
    elite_frac=0.25,
    mutation_rate=0.2,
    weight_ranges=None,        # 新增字段：每个字段的权重范围
    threshold_ranges=None,     # 新增字段：每个字段的阈值范围
    whole_t_range=(0.5, 0.85),
    precision_threshold=0.95,
    negative_pairs=None
):
    weight_ranges = weight_ranges or {f: (0.01, 0.25) for f in fields}
    threshold_ranges = threshold_ranges or {f: (0.5, 0.95) for f in fields}

    population = []
    for _ in range(pop_size):
        weights, thresholds, wthresh = random_config(fields, weight_ranges, threshold_ranges, whole_t_range)
        fitness = fitness_function(weights, thresholds, wthresh, similarity_func, table, ids, ground_truth_pairs,negative_pairs=negative_pairs)
        population.append((fitness, weights, thresholds, wthresh))

    for gen in range(generations):
        population.sort(reverse=True, key=lambda x: x[0])
        best_fitness = population[0][0]
        print(f"============== 第{gen+1}代最大适应度: {best_fitness:.4f} ==============")
        if best_fitness >= precision_threshold:
            print(f"达到精度阈值 {precision_threshold}，提前收敛。")
            break

        elites = population[:max(1, int(pop_size * elite_frac))]
        new_gen = elites.copy()

        while len(new_gen) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child_weights = {f: (p1[1][f] + p2[1][f]) / 2 for f in fields}
            s = sum(child_weights.values())
            child_weights = {f: w / s for f, w in child_weights.items()}

            child_thresholds = {f: round((p1[2][f] + p2[2][f]) / 2, 2) for f in fields}
            child_wthresh = round((p1[3] + p2[3]) / 2, 2)

            if random.random() < mutation_rate:
                field = random.choice(fields)
                low, high = threshold_ranges.get(field, (0.5, 0.95))
                child_thresholds[field] = round(random.uniform(low, high), 2)
                # 重新归一化
                s = sum(child_weights.values())
                child_weights = {f: w / s for f, w in child_weights.items()}

            fitness = fitness_function(child_weights, child_thresholds, child_wthresh, similarity_func, table, ids, ground_truth_pairs,negative_pairs=negative_pairs)
            new_gen.append((fitness, child_weights, child_thresholds, child_wthresh))

        population = new_gen

    best = max(population, key=lambda x: x[0])
    return {
        "fitness": best[0],
        "weights": best[1],
        "thresholds": best[2],
        "whole_threshold": best[3]
    }

# ------------------- 主流程整合 ------------------- #
def main(table="sh_results", 
         ids=None,
         optimize=False,
         ground_truth_pairs=None,
         negative_pairs=None
         ):
    try:
        fields = ["scene_name", "org_name", "contact_name", "ref_title",
                  "constr_background", "constr_necessity", "constr_target",
                  "constr_process", "achievement"]

        if optimize:
            weight_ranges = {
                "scene_name": (0.05, 0.3),
                "org_name": (0.01, 0.05),
                "contact_name": (0.01, 0.05),
                "ref_title": (0.1, 0.3),
                "constr_background": (0.05, 0.2),
                "constr_necessity": (0.05, 0.3),
                "constr_target": (0.1, 0.3),
                "constr_process": (0.1, 0.3),
                "achievement": (0.1, 0.3)
            }

            threshold_ranges = {
                "scene_name": (0.6, 0.9),
                "org_name": (1, 1),
                "contact_name": (1, 1),
                "ref_title": (0.6, 0.9),
                "constr_background": (0.6, 0.9),
                "constr_necessity": (0.6, 0.9),
                "constr_target": (0.6, 0.9),
                "constr_process": (0.6, 0.9),
                "achievement": (0.6, 0.9)
            }

            best_config = genetic_optimize(
                table=table,
                ids=ids,
                ground_truth_pairs=ground_truth_pairs,
                fields=fields,
                similarity_func=calculate_similarity,
                pop_size=100,
                generations=30,
                elite_frac=0.2,
                mutation_rate=0.25,
                weight_ranges=weight_ranges,
                threshold_ranges=threshold_ranges,
                whole_t_range=(0.5, 0.9),
                precision_threshold=0.999,
                negative_pairs=negative_pairs
            )

            weights = best_config["weights"]
            thresholds = best_config["thresholds"]
            whole_threshold = best_config["whole_threshold"]
            print("✅ 自动率定完成，最优参数如下：")
            print(best_config)

        weighted_similarities, matching_pairs = calculate_similarity(
            table, weights, whole_threshold, thresholds, ids
        )

        if weighted_similarities:
            df_reason, dfs = fetch_similarity_details_from_mysql(
                table=table,
                whole_weighted_similarities=weighted_similarities,
                matching_pairs=matching_pairs,
                db_config_path='./input/config.ini'
            )
            save_to_csv(df_reason, dfs)
            return df_reason, dfs
        else:
            print("❌ 没有相似记录")
            return None, None
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        return None, None

# ------------------- 执行入口 ------------------- #
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("./样本数据/样本0707.xlsx",sheet_name="样本")
    unique_names = df['重复标记'].drop_duplicates()
    unique_ids = df['场景ID'].drop_duplicates().to_list()
    unique_split_list = []
    for name in unique_names:
        split_values = name.split("_")
        unique_split_list.append(tuple(split_values))
    
    ground_truth_pairs = [('1116','2131'),('1410','672'),('1443','1977'),('1452','1605'),('1452','2012'),
                        ('1480','1772'),('1480','360'),('1480','366'),
                        ('1498','1642'),('1498','834'),('1498','991'),
                        ('1605','2012'),('1642','834'),('1642','991'),
                        ('1705','171'),('1772','360'),('1772','366'),
                        ('2053','705'),('360','366'),('834','991')]
    positive_pairs = unique_split_list + ground_truth_pairs
    # 负样本
    df1 = pd.read_excel("./样本数据/样本0707.xlsx",sheet_name="负样本")
    unique_names1 = df1['重复标识'].drop_duplicates()
    unique_ids1 = df1['场景编号'].drop_duplicates().to_list()
    unique_split_list1 = []
    for name in unique_names1:
        split_values = name.split("_")
        unique_split_list1.append(tuple(split_values))

    ids = [1116,2131,360,366,1480,1772,1705,171,1977,1443,1410,672,2012,1605,1452,2053,705,1642,991,1498,834]
    ids = unique_ids + unique_ids1 + ids
    ids = list(set(ids))
    main(table="sh_results",
         ids=ids, 
         optimize=True,
         ground_truth_pairs=positive_pairs,
         negative_pairs=unique_split_list1)
