from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from typing import List, Optional, Dict, Any
import aiohttp
import asyncio
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class CustomLLM(BaseChatModel):
    base_url: str = Field(...)
    model_name: str = Field(...)
    api_key: str = Field(...)
    stream: bool = False
    temperature: float = 0.95
    top_p: float = 0.7

    def convert_message(self, m: BaseMessage) -> Dict[str, str]:
        role = "user" if isinstance(m, HumanMessage) else \
               "assistant" if isinstance(m, AIMessage) else \
               "system" if isinstance(m, SystemMessage) else "user"
        return {"role": role, "content": m.content}

    async def call_api(self, messages: List[BaseMessage]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [self.convert_message(m) for m in messages],
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": 0
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise ValueError(f"请求失败，状态码: {resp.status}, 内容: {await resp.text()}")

                if self.stream:
                    full_content = ""
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            try:
                                data = json.loads(line[len("data:"):])
                                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_content += delta
                            except Exception:
                                continue
                    return full_content
                else:
                    data = await resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        return asyncio.run(self.call_api(messages))

    async def _async_call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                          run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        return await self.call_api(messages)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        content = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                         run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        content = await self._async_call(messages, stop=stop, run_manager=run_manager, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
