"""
基于指定权重、阈值、列阈值计算向量数据库chroma中数据的相似度
并将相似度大于阈值的记录进行分组
并从mysql中提取数据
并将相似记录数据保存到mysql中
如果指定ids，则只计算这些ids的相似度
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from util.ReadSave import save_to_csv
from mysql_info import read_config, execute_sql
from vector_storage import get_data_from_chroma
from mysql_info import ShResults, ShResultsP
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
                print("旧权重",pair_weight)
                pair_weight[mask] += add_weight / count
                
                pair_weight[~mask] = 0
                print("新权重",pair_weight)
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


def fetch_similarity_details_from_mysql(table, whole_weighted_similarities, matching_pairs, db_config_path: str):
    """
    通过 id 从 MySQL 中提取数据，并构建重复记录解释表和按列分类的内容对。

    :param table: 表名，如 'sh_results'
    :param whole_weighted_similarities: 所有加权相似度超过阈值的记录 [(id1, id2, similarity)]
    :param matching_pairs: 按字段分类的高相似度对，例如 {'title': [(id1, id2, sim), ...]}
    :param db_config_path: 数据库配置文件路径
    :return: (df_reason, dfs) 
             df_reason 是一个 DataFrame，包含重复对、原因等信息；
             dfs 是一个 dict，按字段名分类的内容对。
    """
    # 1. 提取所有涉及到的 id，避免多次数据库查询
    all_ids = set()
    for id1, id2, _ in whole_weighted_similarities:
        all_ids.update([id1, id2])

    # 2. 加载数据库配置 & 查询数据
    db_config = read_config(db_config_path)
    model = {'sh_results': ShResults,'sh_results_p': ShResultsP}.get(table)

    try:
        df = execute_sql(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            model=model,
            ids=all_ids
        )
    except Exception as e:
        print(f"查询 MySQL 失败: {e}")
        return pd.DataFrame(), {}

    if df.empty:
        print("未查询到任何数据。")
        return pd.DataFrame(), {}

    # 提速：将 id 设置为 index
    df.set_index('id', inplace=True)

    # 3. 构建重复记录说明表
    df_reason = pd.DataFrame(columns=['duplicate_id', 'similarity', 'id', 'scene_name', 'reason'])
    seen_pairs = {}

    for keyword, pairs in matching_pairs.items():
        for id1, id2, _ in pairs:
            key = frozenset((id1, id2))
            duplicate_id = f"{min(id1, id2)}_{max(id1, id2)}"

            # 查找加权相似度记录
            similarity = next(
                (sim for a, b, sim in whole_weighted_similarities if
                 frozenset((a, b)) == key),
                None
            )

            # 查重：如果重复对已存在，就追加原因字段
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
                if rows:
                    df_reason = pd.concat([df_reason, pd.DataFrame(rows)], ignore_index=True)
                seen_pairs[key] = len(df_reason) - 2  # 记录第一行位置

    # 4. 按字段提取内容，构建原始内容对
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

def main(table="sh_results", weights=None, whole_threshold=None, thresholds=None, ids=None):
    """
    主函数，执行相似度计算和数据存储的流程。
    
    :param table: 表名，如 'sh_results'，默认值为 'sh_results'。
    :param weights: 每个字段的权重字典，默认值为 None。
    :param whole_threshold: 整体相似度阈值，默认值为 None。
    :param thresholds: 各个字段的相似度阈值字典，默认值为 None。
    :param ids: 要进行相似度计算的记录 ID 列表，默认值为 None。
    
    :return: 返回重复记录的详细信息表（df_reason）和按字段分类的内容对（dfs）。
    """
    try:
        # 执行相似度计算
        weighted_similarities, matching_pairs = calculate_similarity(
            table, weights, whole_threshold, thresholds, ids
        )

        if weighted_similarities:
            # 从 MySQL 获取相似记录详细信息
            df_reason, dfs = fetch_similarity_details_from_mysql(
                table=table,
                whole_weighted_similarities=weighted_similarities,
                matching_pairs=matching_pairs,
                db_config_path='./input/config.ini'
            )
            save_to_csv(df_reason, dfs)
            return df_reason, dfs
        else:
            print("没有相似记录")
            return None, None
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        return None, None
        
        # if not ids:
        #     print("保存数据到mysql")
        #     save_to_mysql(df, df_reason, dfs, data)

if __name__ == "__main__":
    data = {
        "weights": {
            "scene_name": 0.14,
            "org_name": 0.02,
            "contact_name": 0.02,
            "ref_title": 0.16,
            "constr_background": 0.1,
            "constr_necessity": 0.12,
            "constr_target": 0.14,
            "constr_process": 0.15,
            "achievement": 0.15,
            "scene_overview": 0.5
        },
        "thresholds": {
            "scene_name": 69,
            "org_name": 100,
            "contact_name": 100,
            "ref_title": 69,
            "constr_background": 65,
            "constr_necessity": 61,
            "constr_target": 67,
            "constr_process": 72,
            "achievement": 68,
            "scene_overview": 69
        },
        "whole_threshold": 56
            }  
    weights = data["weights"]
    thresholds = {k: v / 100 for k, v in data["thresholds"].items()}
    whole_threshold = data["whole_threshold"] / 100
    table = "sh_results" # 数据表名
    # table = "sh_results_p"  # 数据表名
    # ids=pd.read_excel('./output/new.xls')['qwe'].to_list()
    ids = [2401,61]
    main(table,weights,whole_threshold,thresholds,ids=ids)
    # main(table,weights,whole_threshold,thresholds)
