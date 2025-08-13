from src.llm import LLM, LLMFile


class MockLLM(LLM):
    async def chat(self, model: str, system_prompt: str, user_prompt: str, agent="mock-llm", return_json=False) -> str:
        return "mocked response"

    async def chat_with_file(
        self, model: str, system_prompt: str, user_prompt: str, files: list[LLMFile], agent: str, return_json: bool = False
    ) -> str:
        return "mocked response"
