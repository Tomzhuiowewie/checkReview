from inter_pack import CustomLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mysql_info import read_config

db_config = read_config("./input/config.ini", section="model")
openai_api_key = db_config.get('OPENAI_API_KEY')
openai_api_base = db_config.get('OPENAI_API_BASE')
model_name = db_config.get('MODEL_NAME')

if not (openai_api_key and openai_api_base and model_name):
    raise ValueError("请检查config.ini文件中 OPENAI_API_KEY / OPENAI_API_BASE / MODEL_NAME 配置")

# 缓存模型与链
llm = CustomLLM(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model_name=model_name,
    temperature=0.01,
    top_p=0.1,
    stream=False
    )


# 构建一次 prompt 和链
veto_prompt = PromptTemplate.from_template(
    """
    # 任务
    判断{result}中是否明确提到“项目成果在本单位、部门或者其他单位、部门的存在实际应用(非成果成效)。”
    # 输出
    输出格式要求：只能输出0或1，不允许多余解释
    # 限制
    1. 判断必须依据文本中是否包含明确描述“应用地点”或“应用状态”的词语，例如“已在本单位应用”、“已投入使用”、“已上线运行”等。
    2. 若文本只描述了某单位作为主体“构建”或“推广”了某项成果（但未直接言明应用的具体地点是某单位，则应视为**未明确提及**。
    3. 严禁根据动作的主体自行推理其应用地点。
    """

)

output_parser = StrOutputParser()
veto_chain = veto_prompt | llm | output_parser
