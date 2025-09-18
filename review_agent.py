from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json,pprint,re
import warnings
import asyncio
from typing import Dict
from prompts import WHOLE_PROMPTS,PROMPTS_CONTAINER
from mysql_info import read_config
from inter_pack import CustomLLM
warnings.filterwarnings("ignore")

db_config = read_config('./input/config.ini', section='model')
openai_api_key = db_config.get('OPENAI_API_KEY')
openai_api_base = db_config.get('OPENAI_API_BASE')
model_name = db_config.get('MODEL_NAME')

# 检查模型配置
if not (openai_api_key and openai_api_base and model_name):
    raise ValueError("请检查config.ini文件模型配置")

llm = CustomLLM(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model_name=model_name,
    temperature=0.01,
    top_p=0.1,
    stream=False
    )

# 定义提示词模版
prompt_template = PromptTemplate.from_template(f"""{WHOLE_PROMPTS}""")

# 定义输入数据格式
class ReviewOut(BaseModel):
    rule: str = Field(description="规则名称")
    score: str = Field(description="评分")
    reason: str = Field(description="评分理由")

# 设置解析器+将指令注入提示模板。
output_parser = JsonOutputParser(pydantic_object=ReviewOut)

# 创建一个信号量，比如最大并发量是5
semaphore = asyncio.Semaphore(100)

def normalize_scores(data):
    """处理模型输出score结果格式不统一问题"""
    score = data.get('score')
    if isinstance(score, str):
        # 先看是不是 N/A
        if score.strip().upper() == "N/A":
            data['score'] = "N/A"
        else:
            # 提取出数字
            match = re.search(r'\d+', score)
            if match:
                data['score'] = int(match.group())
            else:
                data['score'] = "N/A"  # 如果异常，就设成 "N/A"
    return data

def extract_json_substring(text: str) -> str:
    """
    从包含模型多余输出的字符串中提取首个有效 JSON 块
    """
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return match.group()
    except Exception:
        pass
    return None

async def review_scene(idx: int, total: int, new_scene_info: Dict, rule_name: str, rule_prompt: str, score: str) -> Dict:
    chain = prompt_template | llm | output_parser
    new_scene_json = json.dumps(new_scene_info, ensure_ascii=False, separators=(',', ':'))

    for attempt in range(3):
        try:
            async with semaphore:  # 控制并发
                result = await asyncio.wait_for(
                    chain.ainvoke({
                        "score": score,
                        "new_scene": new_scene_json,
                        "rule_name": rule_name,
                        "rule_prompt": rule_prompt,
                    }),
                    timeout=3000,
                )
            
                print(f"[进度] 第{idx}/{total}条 '{rule_name}' 评审完成")
                return normalize_scores(result)
            
        except Exception as e:
            print(f"\n[错误] 第{idx}/{total}条 '{rule_name}' 失败: {e} (重试 {attempt+1}/3)")
            clean_json_str = extract_json_substring(str(e))
            if clean_json_str:
                try:
                    clean_json_str = re.sub(r'\"(\w+)\":\s*([^\s\",{}][^,\n{}]*)', r'"\1": "\2"', clean_json_str)
                    result_dict = json.loads(clean_json_str)
                    print(f"[警告] 第{idx}/{total}条 '{rule_name}' 报错中成功提取 JSON")
                    return normalize_scores(result_dict)
                except Exception as json_err:
                    print(f"[二次失败] JSON解析失败: {json_err}")

            print(f"[错误] 第{idx}/{total}条 '{rule_name}' 失败: {e} (重试 {attempt+1}/3)")
            await asyncio.sleep(1)
        await asyncio.sleep(1)

    print(f"[失败] 第{idx}/{total}条 '{rule_name}' 评审最终失败")
    return {
           "rule": rule_name,
           "score": "错误",
           "reason": "模型调用失败",
       }


if __name__ == "__main__":
    import time

    new_scene_info = {
        "场景名称":"更改营销和图形侧台户关系机器人",
        "解决问题":"解决营销与图形台户关系异常问题",
        "建设必要性":"每天营销系统新装、更名、调户等流程使系统产生大量台户关系异常数据，同时台户关系异常等存量问题剩余较多，公司数据治理指标面临一个巨大挑战，一体化线损受到影响，需要一个智能化机器人维护用户异常地址。",
        "建设目标":"完成全省存量问题数据，每日对新增异动工单进行维护，为理论提供一致的系统档案，为同源系统提供更准确的户变关系模型，方便开展基层开展数据治理工作，提升优质服务。",
        "建设过程":"根据营销专业人员现状向自主研发上线“更改营销和图形侧台户关系机器人”，供电所人员只需导出存量和新增问题清单，填写模板上传，更改营销和图形侧台户关系机器人自动维护流程，协助工作人员快速，高效处理问题数据，省去原先人每条流程工手动发起维护的过程，每条流程节约5-10分钟，机器人反复工作运行至结束，极大的提高工作效率。",
        "成果成效":"该机器人自动维护系统繁琐流程，帮助基层处理问题，减少人工成本和运维时间。截止目前，已累计处理台户关系异常问题数据3000条，每天可节约运维人员成本1小时，一体化线损指标环比23年提升2.3%，电网一张图指标持续保持在99.96%左右，连续5个月排名省公司前20名，洛南公司24年6月荣获百强县公司，八里、寺耳等7家供电所累计12次入选“国网百强供电所”称号，营配“站-线-变-台-户”数据贯通率持续保持在99.99%以上，方便开展基层开展数据治理工作，为客户提供更优质的服务，提高工作效率。"
    }

    async def run_all_reviews():
        results = []
        total = len(PROMPTS_CONTAINER)
        tasks = []

        for idx, (rule_name, rule_prompt) in enumerate(PROMPTS_CONTAINER.items(), start=1):
            if not rule_prompt[0].strip():
                results.append({"rule": rule_name, "score": "N/A", "reason": "该评审规则暂不适用"})
            else:
                task = review_scene(
                    idx=idx,
                    total=total,
                    new_scene_info=new_scene_info,
                    rule_name=rule_name,
                    rule_prompt=rule_prompt[0],
                    score=rule_prompt[1]
                )
                tasks.append(task)

        completed = await asyncio.gather(*tasks)
        results.extend(completed)
        pprint.pprint(results)

    # 启动异步主程序
    asyncio.run(run_all_reviews())
