from fastapi import HTTPException
from mistralai import Mistral as MistralApi, UserMessage, SystemMessage
import logging
from src.session.file_uploads import get_file_content_for_filename, set_file_content_for_filename
from src.utils.file_utils import extract_text
from src.utils import Config
from .llm import LLM, LLMFile
from openai import NOT_GIVEN, AsyncOpenAI, OpenAIError
from typing import List

logger = logging.getLogger(__name__)
config = Config()


class LmStudioIntent(LLM):
    async def chat(self, model, system_prompt: str, user_prompt: str, return_json=False) -> str:
        logger.debug(
            "##### Called lm studio chat ... llm. Waiting on response model with prompt {0}.".format(
                str([system_prompt, user_prompt])
            )
        )
        try:
            lmStudioClient = AsyncOpenAI(api_key="lm-studio", base_url="http://host.docker.internal:1234/v1")
            response = await lmStudioClient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "strict": "false",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "questions": {
                                    "type": "array",
 				                    "items": {
 					                    "type": "string"
 				                    }
                                }
                            }
                        }
                    }
                } if return_json else NOT_GIVEN,
            )
            content = response.choices[0].message.content
            logger.info(f"LM Studio response: Finish reason: {response.choices[0].finish_reason}, Content: {content}")
            logger.debug(f"Token data: {response.usage}")

            if not content:
                logger.error("Call to LM Studio API failed: message content is None")
                return "An error occurred while processing the request."

            return content
        except Exception as e:
            logger.error(f"Error calling LM Studio model: {e}")
            return "An error occurred while processing the request."

    async def chat_with_file(
        self, model: str, system_prompt: str, user_prompt: str, files: list[LLMFile], return_json=False
    ) -> str:
        try:
            for file in files:
                extracted_content = get_file_content_for_filename(file.filename)
                if not extracted_content:
                    extracted_content = extract_text(file)
                    set_file_content_for_filename(file.filename, extracted_content)
                user_prompt += f"\n\nDocument:\n{extracted_content}"
            return await self.chat(model, system_prompt, user_prompt, return_json)
        except Exception as file_error:
            logger.exception(file_error)
            raise HTTPException(status_code=500, detail=f"Failed to process files: {file_error}") from file_error
