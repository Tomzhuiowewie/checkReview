# -*- coding: utf-8 -*-
from mysql_info import read_config
from sentence_transformers import SentenceTransformer
import asyncio
from chromadb import PersistentClient
from transformers import pipeline

class SceneVectorManager:
    def __init__(self, db_path="chroma_db"):

        db_config = read_config('./input/config.ini', section='model')
        embedding_model_name = db_config.get('EMBEDDING_MODEL')
        summary_model = db_config.get('SUMMARY_MODEL')
        if not (embedding_model_name and summary_model):
            raise ValueError("请检查config.ini文件模型配置")
        
        self.chroma_client = PersistentClient(path=db_path)
        self.emb_model = SentenceTransformer(embedding_model_name)
        self.max_length = self.emb_model.max_seq_length
        self.summarizer = pipeline(
            "summarization",
            model=summary_model,
            tokenizer=summary_model
        )

    async def delete_vector(self, scene_id, source='inter'):
        str_id = str(scene_id)
        print(f"准备删除 id: {str_id}, source: {source}")

        def _delete():
            for collection in self.chroma_client.list_collections():
                collection_name = collection.name
                collection = self.chroma_client.get_collection(name=collection_name)
                where = {"source": source} if source else {}
                existing_ids = collection.get(ids=[str_id],where=where)["ids"]
                if existing_ids:
                    print(f"从集合 '{collection_name}' 删除 id: {str_id}")
                    collection.delete(ids=[str_id])
                    print(f"成功删除 id: {str_id}")
                else:
                    print(f"集合 '{collection_name}' 中不存在 id: {str_id}，跳过删除")

        await asyncio.to_thread(_delete)

    async def add_vector(self, new_scene: dict, source='inter'):
        str_id = str(new_scene['id'])
        new_scene_str = {key: str(value) for key, value in new_scene.items()}
        new_scene_emb = {}

        def _add():
            already_exists = False 
            # 检查是否已经存在
            for collection in self.chroma_client.list_collections():
                collection_name = collection.name
                collection = self.chroma_client.get_collection(name=collection_name)
                existing_ids = collection.get(ids=[str_id])["ids"]
                if existing_ids:
                    print(f"场景 ID {str_id} 已存在于集合 '{collection_name}'，跳过添加。")
                    already_exists = True
                    break
            if already_exists:
                return

            for collection in self.chroma_client.list_collections():
                collection_name = collection.name
                col = collection_name.replace("sh_results_", "")
                collection = self.chroma_client.get_collection(name=collection_name)
                content = new_scene_str.get(col, '')
                if len(content) >= self.max_length:
                    summary = self.summarizer(content, max_length=self.max_length, min_length=200, do_sample=False)[0]["summary_text"]
                else:
                    summary = content
                # 非空字段
                non_empty_keys = [key for key, value in new_scene.items() if value]
                # metadatas = [{"source": source, "field": field.get(str(i), "")} for i in processed_ids]
                embedding = self.emb_model.encode(summary)
                collection.add(ids=[str_id], 
                               documents=[summary], 
                               embeddings=[embedding],
                               metadatas=[{"source": source,"field":','.join(non_empty_keys)}])
                new_scene_emb[col] = embedding

            print(f"场景 ID: {str_id} 添加完成")

        await asyncio.to_thread(_add)

    async def update_vector(self, scene_id, update_record, source='inter'):
        print(f"更新场景 ID: {scene_id}")
        await self.delete_vector(scene_id, source)
        await self.add_vector(update_record, source)
        print(f"场景 ID: {scene_id} 更新完成")


if __name__ == "__main__":
    new_scene = {
        "id":"9999",
        "scene_name": "更改营销和图形侧台户关系机器人",
        "ref_title": "解决营销与图形台户关系异常问题",
        "constr_background": "全省115家县公司目前有很多遗留营销与生产侧台区关系不正确存量问题数据，会影响线损指标、营配等各项指标。每天营销系统异动、新装、更名、调户等流程使系统产生大量新增问题数据，这些问题数据考人力维护比较费时、费力，人工维护准确度不高，会产生错误，使用频率高，每天为了线损档案工作统计，都会进行调户。指标要求严，是各级公司同业对标及营销业务重要考核指标之一，目标是全贯通，准确率达到100%，不得出现档案异常用户。",
        "constr_necessity": "每天营销系统新装、更名、调户等流程使系统产生大量台户关系异常数据，同时台户关系异常等存量问题剩余较多，公司数据治理指标面临一个巨大挑战，一体化线损受到影响，需要一个智能化机器人维护用户异常地址。",
        "constr_target": "完成全省存量问题数据，每日对新增异动工单进行维护，为理论提供一致的系统档案，为同源系统提供更准确的户变关系模型，方便开展基层开展数据治理工作，提升优质服务。",
        "constr_process": "根据营销专业人员现状向自主研发上线“更改营销和图形侧台户关系机器人”，供电所人员只需导出存量和新增问题清单，填写模板上传，更改营销和图形侧台户关系机器人自动维护流程，协助工作人员快速，高效处理问题数据，省去原先人每条流程工手动发起维护的过程，每条流程节约5-10分钟，机器人反复工作运行至结束，极大的提高工作效率。",
        "achievement": "该机器人自动维护系统繁琐流程，帮助基层处理问题，减少人工成本和运维时间。截止目前，已累计处理台户关系异常问题数据3000条，每天可节约运维人员成本1小时，一体化线损指标环比23年提升2.3%，电网一张图指标持续保持在99.96%左右，连续5个月排名省公司前20名，洛南公司24年6月荣获百强县公司，八里、寺耳等7家供电所累计12次入选“国网百强供电所”称号，营配“站-线-变-台-户”数据贯通率持续保持在99.99%以上，方便开展基层开展数据治理工作，为客户提供更优质的服务，提高工作效率。"
    }
    async def main():
        db_manager = SceneVectorManager()
        # await db_manager.add_vector(new_scene)
        await db_manager.delete_vector('9999')
    asyncio.run(main())