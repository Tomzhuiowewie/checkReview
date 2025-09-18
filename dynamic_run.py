# -*- coding: utf-8 -*-
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vector_storage import load_model, get_data_from_chroma
from transformers import pipeline
from mysql_info import read_config
import torch,time
import logging
import os
print(os.cpu_count())
import threading

import os
os.environ["CHROMA_DB_MAX_THREADS"] = "6"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_embedding_model():
    db_config = read_config('./input/config.ini', section='model')
    embedding_model = db_config.get('EMBEDDING_MODEL')
    summary_model = db_config.get('SUMMARY_MODEL')
    if not (embedding_model and summary_model):
        raise ValueError("请检查config.ini文件模型配置")
    emb_model = load_model(embedding_model)
    return emb_model, summary_model

emb_model, summary_model = get_embedding_model()
summarizer = pipeline("summarization", model=summary_model)
import os

try:
    available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for platforms without sched_getaffinity
    available_cpus = os.cpu_count()

print(f"[调试] 容器可用线程数: {available_cpus}")

os.environ["CHROMA_DB_MAX_THREADS"] = "6"
chroma_client = chromadb.PersistentClient(path="chroma_db")

logger = logging.getLogger("dup_check")
logger.setLevel(logging.INFO)

def log_step(step_name):
    logger.info(f"[流程追踪] Step: {step_name} | 时间: {time.time():.2f} | "
                f"线程数: {threading.active_count()} | CPU: {os.cpu_count()}")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def dynamic_duplicate_check(data, table, weights, threshold, thresholds, ids=None, source=None):

    logger.info(f"[调试] 当前线程数: {threading.active_count()}")
    log_step("START")
    max_length = emb_model.max_seq_length

    # 所有可能字段
    ALL_FIELDS = list(weights.keys())
    F = len(ALL_FIELDS)
    field_index = {f: i for i, f in enumerate(ALL_FIELDS)}

    # Step 0: 删除空字段
    data = {k: v for k, v in data.items() if v not in (None, "", [], {}, '')}
    log_step("过滤空字段")
    # Step 1: 摘要处理超长字段
    for key, value in data.items():
        if len(value) >= max_length:
            
            summary = summarizer(value,
                                 max_length=int(max_length * 0.8),
                                 min_length=200,
                                 do_sample=False)[0]["summary_text"]
            data[key] = summary
    log_step("摘要长字段")
    # Step 2: 新数据向量化（只保留权重里有的字段）
    encoded_data = {
        k: emb_model.encode(v)
        for k, v in data.items()
        if k in weights and v
    }
    log_step("新样本向量化")
    # Step 3: 获取要比较的历史记录ID
    if ids is None:
        ids = get_data_from_chroma(f"{table}_scene_name", source=source)["ids"]
    N = len(ids)
    log_step(f"获取历史记录 N={N}")
    # Step 4: 获取历史数据向量和元数据
    all_embeddings = {}
    all_metadata = []
    for field in ALL_FIELDS:
        result = get_data_from_chroma(f"{table}_{field}", ids=ids, source=source)
        all_embeddings[field] = result.get("embeddings", [])
        if not all_metadata and "metadatas" in result:
            all_metadata = result["metadatas"]
    log_step("加载历史向量")
    # Step 5: 构建历史 presence 矩阵
    presence_matrix = np.zeros((N, F), dtype=np.uint8)
    for i, meta in enumerate(all_metadata):
        field_str = meta.get("field", "")
        fields_in_meta = [f.strip() for f in field_str.split(",") if f.strip()]
        for f in fields_in_meta:
            if f in field_index:
                presence_matrix[i, field_index[f]] = 1
    log_step("构建新样本")
    # Step 6: 构建新样本 presence
    new_presence = np.zeros(F, dtype=np.uint8)
    for f in encoded_data.keys():
        if f in field_index:
            new_presence[field_index[f]] = 1
    log_step("相似度矩阵")
    # Step 7: 相似度矩阵（历史 N × F）
    similarity_matrix = np.zeros((N, F))
    for f in encoded_data:
        idx = field_index[f]
        new_vec = encoded_data[f].reshape(1, -1)
        old_vecs = all_embeddings.get(f, [])
        if len(old_vecs) != N:
            continue
        sims = cosine_similarity(new_vec, np.stack(old_vecs)).flatten()
        sims = np.where(sims >= thresholds.get(f, 0), sims, 0.0)
        similarity_matrix[:, idx] = sims
    log_step("构建双向有效字段掩码")
    # Step 8: 构建双向有效字段掩码（新样本与历史记录交集）
    adjusted_weights = np.zeros((N, F), dtype=np.float32)
    original_weight_vector = np.array([weights[f] for f in ALL_FIELDS])
    log_step("循环")
    for i in range(N):
        # 历史记录的 presence
        hist_present = presence_matrix[i]
        combined_present = hist_present & new_presence  # 两者都存在的字段
        present_mask = combined_present.astype(bool)

        present_count = present_mask.sum()
        if present_count == 0:
            continue

        # 缺失字段的权重
        missing_mask = ~present_mask
        missing_weight = np.sum(original_weight_vector[missing_mask])

        adjusted_weights[i] = original_weight_vector.copy()
        adjusted_weights[i][present_mask] += missing_weight / present_count
        adjusted_weights[i][~present_mask] = 0  # 缺失字段归零
    log_step("加权相似度")
    # Step 9: 加权相似度
    weighted_similarity = similarity_matrix * adjusted_weights
    total_weights = np.sum(adjusted_weights, axis=1) + 1e-8
    total_scores = np.sum(weighted_similarity, axis=1) / total_weights

    # Step 10: 过滤超阈值
    matched_idx = np.where(total_scores > threshold)[0]
    if len(matched_idx) == 0:
        return {}

    # Step 11: 输出匹配结果
    results = {}
    for i in matched_idx:
        id_ = ids[i]
        cols = {}
        for f, j in field_index.items():
            sim = similarity_matrix[i, j]
            if sim > 0:
                cols[f] = float(round(sim, 2))
        results[id_] = {
            "total_similarity": float(round(total_scores[i], 2)),
            "cols_similarity": cols
        }
    
    print(results)

    return results
