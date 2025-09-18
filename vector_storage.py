"""
本程序实现了以下主要功能：

1. 从 MySQL数据库中读取指定表（如 sh_results）中的结构化数据；
2. 使用指定的中文句向量模型对各字段文本进行嵌入；
3. 利用 Chroma 向量数据库进行向量数据的存储管理，按字段分别创建集合；
4. 对于超出模型最大长度的文本，使用摘要模型进行文本压缩；
5. 支持从 Chroma 中按需读取已有集合中的嵌入内容（如 embeddings、documents、metadatas）；

模块说明：
- mysql_info：自定义模块，用于读取数据库配置和执行查询；
- SentenceTransformer：加载用于生成文本嵌入的模型；
- Chroma：用于向量数据存储与检索；
- Transformers Pipeline：用于摘要生成处理长文本；
- 支持 GPU/CPU 自动切换加载模型。

使用说明：
- 修改 main 函数参数中的数据库配置路径（db_config_path）、模型路径（emb_model_path）和目标表名（table）后运行；
- 向量数据将被按字段分别存储在 Chroma 的集合中，集合命名规则为 “表名_字段名”。

日期：2025-04-11
"""
import os
import chromadb
import torch
from typing import List, Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from mysql_info import read_config, execute_sql, ShResults # 自定义模块
import traceback
from langchain_openai import ChatOpenAI
from pathlib import Path
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 定义一个包装类，将 SentenceTransformer 适配到 ChromaDB 的 EmbeddingFunction 接口
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input, convert_to_numpy=True).tolist()

def load_model(emb_model_path: str) -> Optional[SentenceTransformer]:
    """
    加载 SentenceTransformer 模型。
    :param emb_model_path: 模型路径或名称
    :return: 模型对象
    """
    if not Path(emb_model_path).exists():
        print(f"[错误] 模型路径不存在: {emb_model_path}")
        return None
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(emb_model_path, device=device, trust_remote_code=True)
        return model
    except Exception as e:
        print(f"[错误] 加载模型失败: {e}\n{traceback.format_exc()}")
        return None

def store_by_chrome(
    collection_name: str,
    ids: List[str],
    documents: Optional[List[str]] = None,
    emb_model: Optional[SentenceTransformer] = None,
    summary_model: Optional[str] = None,
    embeddings: Optional[List[List[float]]] = None,
    field: Optional[str] = None,
    source: Optional[str] = 'inter'  # 默认值为 'inter'，可根据需要修改
):

    try:
        # 初始化 Chroma 客户端
        client = chromadb.PersistentClient(path="chroma_db")

        # 参数校验
        if not documents and not embeddings:
            raise ValueError("必须提供 documents 或 embeddings 参数")

        # 检查集合是否存在(如果存在则删除)
        try:
            client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
            print(f"[删除] 已删除 collection: {collection_name}")
        except chromadb.errors.NotFoundError:
            print(f"[跳过] Collection {collection_name} 不存在，跳过删除。")

        # 准备集合
        if emb_model:
            embedding_function = SentenceTransformerEmbeddingFunction(emb_model)
            collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
        else:
            collection = client.create_collection(name=collection_name)

        if documents:
            processed_documents = [] # 处理后的文档列表
            processed_ids = [] # 处理后的 ID 列表（uuid）
            processed_embeddings = [] # 处理后的嵌入向量列表

            summarizer = None
            max_length = emb_model.max_seq_length if emb_model else 512

            # 初始化 summarizer（只初始化一次）
            if emb_model and any(len(doc) >= max_length for doc in documents):
                try:
                    summarizer = pipeline("summarization", model=summary_model)
                    print("[初始化] Summarizer模型已加载")
                except Exception as e:
                    print(f"[警告] Summarizer加载失败，长文本将不会被摘要: {e}")

            # 文档处理，带进度条
            print(f"[开始] 正在处理 {len(documents)} 篇文档...")
            for idx, doc in tqdm(enumerate(documents), total=len(documents), desc="处理文档中"):
                try:
                    content = doc
                    if summarizer and len(doc) >= max_length:
                        summary = summarizer(doc, max_length=int(max_length*0.8), min_length=200, do_sample=False)
                        content = summary[0]["summary_text"]

                    processed_documents.append(content)
                    processed_ids.append(str(ids[idx]))
                    # 编码生成向量
                    if emb_model:
                        embedding = emb_model.encode(content)
                        processed_embeddings.append(embedding)
                except Exception as e:
                    print(f"[警告] 文档索引 {idx} 处理失败: {e}，内容预览: {doc[:50]}...")
                    continue

            # 批量添加到 Chroma
            # metadata 改为每条记录单独获取其 field 字段
            metadatas = [{"source": source, "field": field.get(str(i), "")} for i in processed_ids]

            collection.add(ids=processed_ids, 
                           documents=processed_documents,
                           embeddings=processed_embeddings if emb_model else None,
                           metadatas=metadatas)
        else:
            # 直接用传入的 embeddings
            metadatas = [{"source": source, "field": field.get(str(i), "")} for i in ids] 
            print(f"[开始] 正在存储 {len(embeddings)} 个嵌入向量...")
            collection.add(ids=ids, 
                           embeddings=embeddings,
                           documents=documents,
                           metadatas=metadatas)

        print(f"[完成] 数据已存储到集合 '{collection_name}' 中。")

    except Exception as e:
        print(f"[错误] 存储数据到 Chroma 失败: {e}\n{traceback.format_exc()}")
        raise

def get_data_from_chroma(table,
                         ids=None,
                         inclued_columns=["embeddings", "metadatas", "documents"],
                         source: Optional[str] = None):
    """
    从 Chroma 数据库中直接取出原表向量化内容和聚类中心数据。
    :param table: 表名
    :return: 原表向量化内容和聚类中心数据
    """
    # 初始化 Chroma 客户端，避免每次调用都重新初始化
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    # 取出原表向量化内容
    raw_col_embeddings_collection = chroma_client.get_collection(name=table)
    # 构造 where 筛选条件（只有当传入 source 时才过滤）
    where_filter = {"source": source} if source else {}
    # 获取数据（根据是否提供了 ids 选择查询方式）
    if ids:
        ids = [str(id) for id in ids]
        if where_filter:
            raw_data = raw_col_embeddings_collection.get(ids=ids, include=inclued_columns, where=where_filter)
        else:
            raw_data = raw_col_embeddings_collection.get(ids=ids, include=inclued_columns)
    else:
        if where_filter:
            raw_data = raw_col_embeddings_collection.get(include=inclued_columns, where=where_filter)
        else:
            raw_data = raw_col_embeddings_collection.get(include=inclued_columns)
    print(f"正在从 '{table}' 取出数据...")

    return raw_data

def main(db_config_path: str,
         emb_model_path: str,
         summary_model: str,
         table: str,
         source: str,
         enable_summary: bool = True,
         enable_keywords: bool = True):
    """
    主函数：从 MySQL 拉取数据，向量化存入 Chroma。

    :param db_config_path: 数据库配置文件路径
    :param emb_model_path: 嵌入模型路径
    :param table: 表名
    """
    try:
        if not Path(db_config_path).exists():
            raise FileNotFoundError(f"数据库配置文件不存在: {db_config_path}")
            
        # 从 MySQL 数据库中获取数据
        try:
            db_config = read_config(db_config_path)
            model = {'sh_results': ShResults}.get(table)
            if model is None:
                raise ValueError(f"不支持的表名: {table}")
                
            df = execute_sql(db_config["host"],
                             db_config["port"],
                             db_config["user"],
                             db_config["password"],
                             db_config["database"],
                             model,
                            #  ids=list(range(1, 50)),
                             filters={"is_del": None,"status": 2})
        except Exception as e:
            print(f"[错误] 数据库查询失败: {e}")
            return

        # 加载嵌入模型
        emb_model = load_model(emb_model_path)
        if emb_model is None:
            print("[错误] 嵌入模型加载失败，终止程序")
            return

        df = df.astype(str)  # 转换为字符串类型
        ids = df['uuid'].tolist()
        all_fields = df.columns.tolist()
        fields_list = {}

        for i in range(len(df)):
            row = df.iloc[i]
            id = row['uuid']
            non_empty_fields = []

            for col in all_fields:
                val = row[col].strip()
                if val != 'None' and val != '':  # 非空字符串
                    non_empty_fields.append(col)

            fields_list[str(id)] = ",".join(non_empty_fields)

        for col in df.columns:
            if col == 'id' or col == 'uuid' or col == 'p_up_id':
                print(f"[跳过] 跳过字段: {col}")
                continue
            collection_name = f"{table}_{col}"  # 集合名称
            print(f"[处理] 正在向量化存储字段: {col}")
            store_by_chrome(collection_name=collection_name,
                            ids=ids,
                            documents=df[col].tolist(),
                            emb_model=emb_model,
                            summary_model=summary_model,
                            field=fields_list,
                            source=source)
            print(f"[完成] 字段 {col} 处理完成")
        print("[完成] 所有字段处理完成")

    except Exception as e:
        print(f"[严重错误] 主程序执行失败: {e}\n{traceback.format_exc()}")
        return

if __name__ == "__main__":
    db_config = read_config('./input/config.ini', section='model')
    emb_model_path = db_config.get('EMBEDDING_MODEL')
    summary_model = db_config.get('SUMMARY_MODEL')
    if not (emb_model_path and summary_model):
        raise ValueError("请检查config.ini文件模型配置")
    
    main(
        db_config_path="./input/config.ini",
        emb_model_path=emb_model_path,
        summary_model=summary_model,
        table="sh_results",
        source='inter',
        enable_summary=False,
        enable_keywords=False
    )
