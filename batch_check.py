"""
批量查重
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vector_storage import get_data_from_chroma
from collections import defaultdict

def calculate_similarity(table, weights, whole_threshold, col_threshold_list, ids):
    try:
        # Step0 依据权重决定查重字段
        all_fields = list(weights.keys())
        field_index = {f: i for i, f in enumerate(all_fields)}
        F = len(all_fields)

        # 1. 获取每个字段的向量数据
        all_col_data = {}
        for col in all_fields:
            try:
                all_col_data[col] = get_data_from_chroma(f"{table}_{col}", ids=ids)
            except Exception as e:
                print(f"获取向量失败 - {col}: {e}")
                return None, None

        # 2. 构建 presence 矩阵 (N × F)，记录每条记录存在哪些字段
        sample_col = next(iter(all_col_data))
        ids_list = all_col_data[sample_col]['ids']
        N = len(ids_list)
        if not ids_list:
            print("ID 列表为空。")
            return None, None

        # 2. 构建 presence 矩阵 (N × F)，记录每条记录存在哪些字段
        presence = np.zeros((N, F), dtype=np.uint8)
        for f, i in field_index.items():
            metas = all_col_data[f].get("metadatas", [])
            for j, meta in enumerate(metas):
                if meta and "field" in meta and f in meta["field"]:
                    presence[j, i] = 1

        # 3. 逐字段计算相似度矩阵（过滤低于阈值的值）
        similarity_matrix = np.zeros((N, N, F))
        for f, i in field_index.items():
            embeddings = all_col_data[f].get("embeddings", [])
            # if embeddings is None or not isinstance(embeddings, list) or len(embeddings) != N:
            #     continue

            sims = cosine_similarity(embeddings)
            threshold = col_threshold_list.get(f, 1.0)
            sims = np.where(sims >= threshold, sims, 0.0)
            similarity_matrix[:, :, i] = sims

        # 4. 构建 pair-wise 动态权重矩阵 (N×N×F)
        adjusted_weights = np.zeros((N, N, F), dtype=np.float32)
        original_weight_vec = np.array([weights[f] for f in all_fields])

        for i in range(N):
            for j in range(i + 1, N):
                mask = (presence[i] & presence[j]).astype(bool)  # 双方都存在的字段
                count = np.sum(mask)
                if count == 0:
                    continue
                redistributed = original_weight_vec * (~mask)
                add_weight = redistributed.sum()
                pair_weight = original_weight_vec.copy()
                # print("旧权重",pair_weight)
                pair_weight[mask] += add_weight / count
                
                pair_weight[~mask] = 0
                # print("新权重",pair_weight)
                adjusted_weights[i, j] = pair_weight
                adjusted_weights[j, i] = pair_weight  # 对称

        # 5. 加权合并相似度
        weighted_sim_matrix = np.sum(similarity_matrix * adjusted_weights, axis=2)
        total_weight_matrix = np.sum(adjusted_weights, axis=2) + 1e-8
        final_score_matrix = weighted_sim_matrix / total_weight_matrix

        # 6. 高相似度对筛选
        weighted_similarities = []
        valid_pairs_set = set()  # 用于快速判断合法对

        for i in range(N):
            for j in range(i + 1, N):
                score = final_score_matrix[i, j]
                if score >= whole_threshold:
                    pair = (ids_list[i], ids_list[j])
                    weighted_similarities.append((*pair, round(score, 3)))
                    valid_pairs_set.add(frozenset(pair))  # 用 frozenset 保证无序对的唯一性

        # 7. 匹配字段对保存（为生成字段原因用）
        high_similarity_pairs = defaultdict(list)

        for f, fi in field_index.items():
            sims = similarity_matrix[:, :, fi]
            if sims.shape[0] != N or sims.shape[1] != N:
                print(f"[警告] 字段 {f} 的相似度矩阵尺寸异常，跳过")
                continue
            for m in range(N):
                for n in range(m + 1, N):
                    sim = sims[m, n]
                    pair = frozenset((ids_list[m], ids_list[n]))
                    if sim > 0 and pair in valid_pairs_set:
                        high_similarity_pairs[f].append((ids_list[m], ids_list[n], round(sim, 2)))


        return weighted_similarities, high_similarity_pairs

    except Exception as e:
        print(f"[错误] 计算相似度失败: {e}")
        return None, None
    
if __name__ == "__main__":
    data = {
        "weights": {
            "scene_name": 0.1,
            "ref_title": 0.1,
            "constr_background": 0.05,
            "constr_necessity": 0.1,
            "constr_target": 0.15,
            "constr_process": 0.2,
            "achievement": 0.3
        },
        "thresholds": {
            "scene_name": 69,
            "ref_title": 65,
            "constr_background": 69,
            "constr_necessity": 69,
            "constr_target": 69,
            "constr_process": 69,
            "achievement": 69
        },
        "whole_threshold": 54
            }
    weights = data["weights"]
    thresholds = {k: v / 100 for k, v in data["thresholds"].items()}
    whole_threshold = data["whole_threshold"] / 100
    table = "sh_results"
    ids = ["1523","1582","1700"]

    weighted_similarities, matching_pairs = calculate_similarity(
            table, weights, whole_threshold, thresholds, ids)
