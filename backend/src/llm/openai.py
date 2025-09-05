import asyncio
import logging
import time

from src.utils.power_usage import calculate_power_usage
from src.utils import Config
from src.llm.llm import LLM, LLMFile, LLMFileUploadManager
from src.session.llm_file_upload import (
    add_llm_file_upload,
    get_all_files,
    get_llm_file_upload_id,
    reset_llm_file_uploads,
)
from openai import NOT_GIVEN, AsyncOpenAI, OpenAIError
from openai.types.beta.threads import Text

logger = logging.getLogger(__name__)
config = Config()


def remove_citations(message: Text):
    value = message.value
    for annotation in message.annotations:
        value = value.replace(annotation.text, "")
    return value


class OpenAI(LLM):
    async def chat(self, model, system_prompt: str, user_prompt: str, agent: str, return_json=False) -> str:
        logger.debug(
            "##### Called open ai chat ... llm. Waiting on response model with prompt {0}.".format(
                str([system_prompt, user_prompt])
            )
        )
        try:
            client = AsyncOpenAI(api_key=config.openai_key)
            start_time = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"} if return_json else NOT_GIVEN,
            )
            duration = time.time() - start_time
            content = response.choices[0].message.content

            # Prepare token usage data for logging
            if hasattr(response, "usage") and response.usage is not None:
                token_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            else:
                logger.warning("No usage data in OpenAI response")
                token_info = {
                    "prompt_tokens": "N/A",
                    "completion_tokens": "N/A",
                    "total_tokens": "N/A",
                }

            power_usage = calculate_power_usage(duration, model)

            self.record_usage(
                model=model,
                provider="openai",
                agent=agent,
                token_usage=token_info,
                duration=duration,
                power_usage=power_usage,
            )

            logger.info(f"OpenAI response: Finish reason: {response.choices[0].finish_reason}, Content: {content}")
            logger.info(f"Response Usage: {response.usage}")
            logger.debug(f"Token data: {response.usage}, Duration: {duration:.2f}s")
            logger.debug(f"OpenAI power usage: {power_usage:.2f} Wh")

            if not content:
                logger.error("Call to Open API failed: message content is None")
                return "An error occurred while processing the request."

            return content
        except Exception as e:
            logger.error(f"Error calling OpenAI model: {e}")
            return "An error occurred while processing the request."

    async def chat_with_file(
        self, model: str, system_prompt: str, user_prompt: str, files: list[LLMFile], agent: str, return_json=False
    ) -> str:
        client = AsyncOpenAI(api_key=config.openai_key)
        start_time = time.time()

        file_ids = await OpenAILLMFileUploadManager().upload_files(files)

        vector_store_id = await OpenAILLMFileUploadManager().add_files_to_vector_store(file_ids)

        response = await client.responses.create(
            model=model,
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            text={"format": {"type": "json_object"}} if return_json else NOT_GIVEN,
        )

        message = response.output_text

        duration = time.time() - start_time

        if hasattr(response, "usage") and response.usage is not None:
            token_info = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:
            logger.warning("No usage data in OpenAI File response")
            token_info = {
                "prompt_tokens": "N/A",
                "completion_tokens": "N/A",
                "total_tokens": "N/A",
            }

        power_usage = calculate_power_usage(duration, model)
        # Log to CSV file using base class method
        self.record_usage(
            model=model,
            provider="openai-file",
            agent=agent,
            token_usage=token_info,
            duration=duration,
            power_usage=power_usage,
        )

        logger.info(f"OpenAI file-based response: Message length: {len(message) if message else 0}")
        logger.debug(f"Token usage: {token_info}, Duration: {duration:.2f}s")
        return message


class OpenAILLMFileUploadManager(LLMFileUploadManager):
    async def upload_files(self, files: list[LLMFile]) -> list[str]:
        client = AsyncOpenAI(api_key=config.openai_key)

        file_ids = []
        files_to_upload = []
        start_time = time.time()
        for file in files:
            file_id = get_llm_file_upload_id(file.filename)
            if not file_id:
                logger.info(f"Open AI: Preparing to upload '{file.filename}'")
                file = (file.filename, file.file) if isinstance(file.file, bytes) else file.file
                files_to_upload.append(client.files.create(file=file, purpose="user_data"))
            else:
                file_ids.append(file_id)
                logger.info(f"Open AI: {file.filename} already uploaded to OpenAI with id '{file_id}'")

        uploaded_files = await asyncio.gather(*files_to_upload)

        for file in uploaded_files:
            add_llm_file_upload(file.id, file.filename)
            file_ids.append(file.id)
            logger.info(f"Open AI: File '{file.filename}' uploaded with id '{file.id}'")

        if uploaded_files:
            logger.info(f"Open AI: Time to upload files {time.time() - start_time}")

        return file_ids

    async def add_files_to_vector_store(self, file_ids: list[str]) -> str:
        client = AsyncOpenAI(api_key=config.openai_key)

        vector_store = await client.vector_stores.create(
            name="knowledge_base",
            expires_after={
                "anchor": "last_active_at",
                "days": 1,
            },
        )

        await client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store.id,
            file_ids=file_ids,
        )

        return vector_store.id

    async def delete_all_files(self):
        try:
            client = AsyncOpenAI(api_key=config.openai_key)
            files = get_all_files()
            logger.info(f"Open AI: deleting files {files}")
            delete_tasks = [client.files.delete(file_id=file["file_id"]) for file in files]
            await asyncio.gather(*delete_tasks)
            reset_llm_file_uploads()
            logger.info("Open AI: Files deleted")
        except OpenAIError:
            logger.info("OpenAI not configured")
