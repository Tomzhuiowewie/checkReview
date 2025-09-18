import chromadb

# 本地持久化路径，例如 "./chroma_data"
client = chromadb.PersistentClient(path="./chroma_db")

# 列出所有集合
collections = client.list_collections()
print("Collections:", [c.name for c in collections])

# 加载某个集合，例如 "my_collection"
collection = client.get_collection(name="sh_results_scene_name")

# 查看集合中全部内容（返回最多100条）
results = collection.get(include=["documents", "metadatas", "embeddings"])
print(len(results))
for i, doc in enumerate(results["documents"]):
    # if results['ids'][i] == 1984:
        print(f"{i+1}. Document: {doc}")
        print(f"   ID: {results['ids'][i]}")
        # print(f"   embeddings: {results['embeddings'][i]}")
        print(f"   Metadata: {results['metadatas'][i]}")