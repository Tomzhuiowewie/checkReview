import logging
from llm_client import veto_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import json
import uvicorn, ast, asyncio, os,re,time
from batch_check import calculate_similarity
from dynamic_run import dynamic_duplicate_check
from DBManger import  SceneVectorManager
from review_agent import review_scene
from fastapi.middleware.cors import CORSMiddleware
from prompts import PROMPTS_CONTAINER
from report_gen import generate_report
from mysql_info import read_config
from inter_pack import CustomLLM
import psutil
from pynvml import *

# ================ 日志配置 ===================================
logger = logging.getLogger("server_logger")
logger.setLevel(logging.INFO)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

# 文件日志
file_handler = logging.FileHandler("server.log", encoding="utf-8")
file_handler.setLevel(logging.ERROR)
file_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# 防止重复添加 handler
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
# ================ 应用和路由 ==================================
app = FastAPI()
# 允许的来源（前端运行地址）
# origins = [
#     "http://localhost:3000",      # 本地前端开发地址
#     "http://127.0.0.1:3000",
#     "http://your-frontend.com",   # 部署后的前端地址
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 允许的来源
    allow_credentials=True,         # 是否允许发送 cookies
    allow_methods=["*"],            # 允许所有请求方法（GET、POST等）
    allow_headers=["*"],            # 允许所有请求头
)
# ====================== 通用工具函数 ==========================
def parse_ids(ids: Optional[Any]) -> Optional[List[int]]:
    if ids:
        return ast.literal_eval(ids) if isinstance(ids, str) else ids
    return None

# ======================    资源监测   ==========================
async def sample_resources(interval: float, samples: dict, stop_event: asyncio.Event):
    """异步循环采样CPU、内存、GPU使用率，存入samples，直到stop_event被设置"""
    try:
        nvmlInit()
    except Exception as e:
        logger.warning(f"NVML初始化失败: {e}")
        nvml_available = False
    else:
        nvml_available = True

    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=None)  # 非阻塞采样
        mem = psutil.virtual_memory().used / 1024 / 1024  # MB

        gpu_utils = []
        gpu_mem_used = []
        if nvml_available:
            try:
                device_count = nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    gpu_utils.append(util.gpu)
                    gpu_mem_used.append(mem_info.used / 1024 / 1024)
            except Exception as e:
                logger.warning(f"采样GPU数据失败: {e}")

        # 更新峰值
        samples["cpu"] = max(samples.get("cpu", 0), cpu)
        samples["mem"] = max(samples.get("mem", 0), mem)

        if gpu_utils:
            if "gpu_utils" not in samples:
                samples["gpu_utils"] = gpu_utils
                samples["gpu_mem"] = gpu_mem_used
            else:
                samples["gpu_utils"] = [max(a, b) for a, b in zip(samples["gpu_utils"], gpu_utils)]
                samples["gpu_mem"] = [max(a, b) for a, b in zip(samples["gpu_mem"], gpu_mem_used)]

        await asyncio.sleep(interval)

    if nvml_available:
        try:
            nvmlShutdown()
        except Exception:
            pass

def format_resource_str(res):
    gpu_str = " | ".join(
        f"GPU {i}: {u}% | {m:.1f} MB"
        for i, (u, m) in enumerate(zip(res.get("gpu_utils", []), res.get("gpu_mem", [])))
    ) if res.get("gpu_utils") else "GPU info not available"
    return f"CPU: {res.get('cpu', 0):.1f}% | Mem: {res.get('mem', 0):.1f} MB | {gpu_str}"

@app.middleware("http")
async def resource_peak_middleware(request: Request, call_next):
    logger.info(f"[请求开始] {request.method} {request.url.path}")

    samples = {}
    stop_event = asyncio.Event()
    sampling_task = asyncio.create_task(sample_resources(0.2, samples, stop_event))

    start_time = time.time()
    try:
        response = await call_next(request)
    finally:
        # 请求结束，停止采样
        stop_event.set()
        await sampling_task

    duration = (time.time() - start_time) * 1000

    logger.info(f"[资源监控-峰值] {format_resource_str(samples)}")
    logger.info(f"[请求结束] {request.method} {request.url.path} 耗时: {duration:.1f}ms")
    return response

# ====================== Pydantic 模型 ==========================
class CheckConfig(BaseModel):
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    whole_threshold: float

class SceneContent(BaseModel):
    class Config:
        extra = "allow"

class NewSceneCheckInput(BaseModel):
    scene: SceneContent
    config: CheckConfig
    source: Optional[str] = None

class BatchCheck(BaseModel):
    ids: List[Any]
    config: CheckConfig

class SceneRecordRequest(BaseModel):
    new_scene: Dict[str, Any] = None
    type: str
    delete_scene_id: Optional[Any] = None
    source: Optional[str] = None

# ======================== 接口部分 =============================
@app.post("/api/new_scene_check/")
async def new_scene_check(input: NewSceneCheckInput):
    logger.info("[new_scene_check] 接收到新场景数据，开始查重...")
    try:
        weights = {k:v / 100 for k, v in input.config.weights.items()}
        thresholds = {k: v / 100 for k, v in input.config.thresholds.items()}
        whole_threshold = input.config.whole_threshold / 100
        scene_data = input.scene.model_dump()
        source = input.source  # 获取 source 字段，默认为None

        filter_data = {k: v for k, v in scene_data.items() if k in weights and weights[k] != 0}

        if not filter_data:
            raise HTTPException(status_code=400, detail="缺少有效字段用于查重")

        results = dynamic_duplicate_check(
            data=filter_data,
            table="sh_results",
            weights=weights,
            thresholds=thresholds,
            threshold=whole_threshold,
            ids=None,
            source=source
        )

        if results:
            logger.info(f"[完成] 查重结果：\n{json.dumps(results, indent=2, ensure_ascii=False)}")
        return results if results else {"message": "没有找到与该场景相关的相似记录"}

    except Exception as e:
        logger.error("[new_scene_check] 查重异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/new_scene_review/")
async def new_scene_review(new_scene: SceneContent):
    logger.info("[new_scene_review] 开始新场景评审...")
    try:
        input_data = new_scene.model_dump()
        tasks = []
        rules = list(PROMPTS_CONTAINER.items())
        total = len(rules)

        for idx, (rule_name, rule_prompt) in enumerate(rules, 1):
            if rule_prompt[0]:
                tasks.append(
                    review_scene(
                        idx, total, input_data, rule_name, rule_prompt[0], rule_prompt[1]
                    )
                )
            else:
                logger.info(f"[跳过] 第{idx}/{total}条 '{rule_name}' 无评审内容，跳过。")
                tasks.append(
                    asyncio.sleep(0, result={
                        "rule": rule_name,
                        "score": "N/A",
                        "reason": "此项暂不支持AI评审"
                    })
                )
        # 启动任务
        results = await asyncio.gather(*tasks)
        logger.info(f"[完成] 所有{total}条评审任务")
        return results

    except Exception as e:
        logger.error("[new_scene_review] 评审异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/modify_scene_record/")
async def modify_scene_record(request: SceneRecordRequest):  # 类型: "add", "delete", update"
    logger.info("[modify_scene_record] 开始保存新场景记录...")
    
    new_scene = request.new_scene
    delete_id = request.delete_scene_id
    type = request.type
    source = request.source
    try:
        # 先做输入校验
        if type not in {"add", "delete", "update"}:
            raise HTTPException(status_code=400, detail="参数type必须是 'add'、'delete' 或 'update'")
        # 根据type的值选择对应的操作
        if type == "add":
            logger.info(f"[modify_scene_record] 准备新增场景{new_scene}")
            await SceneVectorManager().add_vector(new_scene,source)  # 异步调用
            logger.info("[modify_scene_record] 场景向量已添加")
        elif type == "delete":
            logger.info(f"[modify_scene_record] 准备删除场景{delete_id}")
            await SceneVectorManager().delete_vector(scene_id=delete_id,source=source)  # 异步调用
            logger.info("[modify_scene_record] 场景向量已删除")
        elif type == "update":
            logger.info(f"[modify_scene_record] 准备更新场景：\n删除{delete_id}\n新增{new_scene}")
            await SceneVectorManager().update_vector(delete_id,new_scene,source)  # 异步调用
            logger.info("[modify_scene_record] 场景向量已更新")
        
        return {"message": "场景向量记录操作成功"}
    except Exception as e:
        logger.error("[modify_scene_record] 异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch_check/")
async def batch_check(params: BatchCheck):
    logger.info("[batch_check] 批量查重中，ID列表: %s", params.ids)
    try:
        # 从用户传入的 config 中获取配置
        weights =  {k: v / 100 for k, v in params.config.weights.items()}
        thresholds = {k: v / 100 for k, v in params.config.thresholds.items()}
        whole_threshold = params.config.whole_threshold / 100

        df_reason, dfs = calculate_similarity(
            table="sh_results",
            weights=weights,
            col_threshold_list=thresholds,
            whole_threshold=whole_threshold,
            ids=params.ids
        )

        if df_reason:
            logger.info(f"[完成] 批量查重\n results: {df_reason.to_dict() if hasattr(df_reason, 'to_dict') else df_reason}\n detail: {dfs}"
                        )

        return {
            "results": df_reason.to_dict() if hasattr(df_reason, 'to_dict') else df_reason,
            "detail": dfs
        }
    except Exception as e:
        logger.error("[batch_check] 异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report_gen/")
async def report_gen(request: Dict[str, Any] = Body(...)):
    logger.info("[report_gen] 生成报告中...")
    # 报告类型："review_sum", "ai_analyze".
    report_type = request.get("report_type", "review_sum")
    logger.info(f"[report_gen] 生成{report_type}报告")
    
    db_config = read_config('./input/config.ini', section=f'{report_type}')
    file_path = db_config.get('file_path')
    output_file_path = db_config.get('output_file_path')
    pic_path = db_config.get('pic_path')
    relate_data = request.get("data",{})

    try:
        generate_report(template_path=file_path,
                        relate_data=relate_data,
                        pic_path=pic_path,
                        output_file_path=output_file_path)
        logger.info("[report_gen] 报告生成完成")
         # 返回文件
        return FileResponse(
            path=output_file_path,
            filename=os.path.basename(output_file_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document" 
        )
    except Exception as e:
        logger.error("[report_gen] 异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/veto/")
async def veto_scene(scene: SceneContent, task: str = Body(...)):
    logger.info("[veto_scene] 开始评判...")

    scene_name = scene.scene_name.strip()
    logger.info(f"[veto_scene] 输入场景名称: {scene_name}")
    forbidden = {"平台","工具","系统"}
    if any(word in scene_name for word in forbidden):
        return {"veto_result": "场景名称中含有平台、工具、系统，审查不通过" }

    result = scene.achievement.strip()
    logger.info(f"[veto_scene] 输入成果成效: {result}")

    try:
        summary = await veto_chain.ainvoke({"result": result})

        logger.info(f"[veto_scene] 原始模型输出: {summary}")

        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
        if summary not in {"0", "1"}:
            logger.info("输出结果为非法输出，自动上报成功")
            summary = "1"

        message = (
            "上报成功"
            if summary == "1"
            else "该场景的成果描述中，未体现在本单位或其他单位的试点应用及成果成效"
        )

        logger.info("[完成] 评判结束")
        return {"veto_result": message}

    except Exception as e:
        logger.error("[veto_scene] 异常: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

#　上传文件接口
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, os, zipfile
from pathlib import Path

class FileUploadRequest(BaseModel):
    filename: str
    content_base64: str
    save_path: str = "./"

def is_safe_path(basedir: str, path: str) -> bool:
    return Path(os.path.abspath(path)).resolve().is_relative_to(Path(basedir).resolve())

@app.post("/api/upload_base64/")
async def upload_file(request: FileUploadRequest):
    try:
        # 校验路径合法性
        if not is_safe_path("./", request.save_path):
            raise HTTPException(status_code=400, detail="非法保存路径")

        os.makedirs(request.save_path, exist_ok=True)
        file_path = os.path.join(request.save_path, request.filename.lower())

        # 解码 Base64 文件内容
        try:
            file_bytes = base64.b64decode(request.content_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Base64 解码失败")

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # 解压 zip（如果是）
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    member_path = os.path.abspath(os.path.join(request.save_path, member))
                    if not member_path.startswith(os.path.abspath(request.save_path)):
                        raise HTTPException(status_code=400, detail="压缩包中包含非法路径")
                zip_ref.extractall(request.save_path)
            os.remove(file_path)
            return {
                "message": "压缩包上传成功并解压完成",
                "extracted_to": request.save_path
            }
        else:
            return {
                "message": "文件上传成功",
                "file_path": file_path
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# =========================== 启动入口 ===========================
if __name__ == "__main__":
    try:
        logger.info("服务器启动中...")
        uvicorn.run("server:app", host="0.0.0.0", port=20236, reload=True)
    except Exception as e:
        logger.error("启动服务器时出现错误: %s", str(e), exc_info=True)
        print(f"启动服务器时出现错误: {e}")
