from fastapi import HTTPException
import logging
import json
import aiohttp
import re
import time
from src.utils import Config
from src.session.file_uploads import get_file_content_for_filename, set_file_content_for_filename
from src.utils.file_utils import extract_text
from .llm import LLM, LLMFile

logger = logging.getLogger(__name__)
config = Config()


class LMStudio(LLM):
    """
    LM Studio client that connects to a locally running instance.
    Uses the OpenAI-compatible API provided by LM Studio.
    This implementation uses aiohttp to directly call LM Studio's API endpoints.
    """

    async def chat(self, model, system_prompt: str, user_prompt: str, return_json=False) -> str:
        logger.debug(
            "Called LMStudio llm. Waiting on response with prompt {0}.".format(str([system_prompt, user_prompt]))
        )

        url = config.lmstudio_url
        if url is None:
            logger.error("LMSTUDIO_URL configuration is missing")
            raise ValueError(
                "LMSTUDIO_URL is not configured. Please set this in your environment variables or .env file."
            )

        # Make sure we have a clean URL without trailing slash
        if url.endswith("/"):
            url = url[:-1]

        # Construct the API endpoint
        url = f"{url}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        # If JSON is requested, modify the system prompt to ensure valid JSON response
        if return_json:
            system_prompt = (
                system_prompt + "\nIMPORTANT: You must respond with valid JSON only. Format your entire response as a "
                "proper JSON object."
            )

        payload = {
            "model": model or config.lmstudio_model or "local-model",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0,
            "max_tokens": config.lmstudio_max_tokens,  # Get token limit from config
        }

        # Log the complete payload for debugging
        logger.debug(f"LM Studio API request payload: {json.dumps(payload, indent=2)}")
        logger.info(f"Sending direct HTTP request to LM Studio at {url}")

        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    duration = time.time() - start_time
                    logger.debug(f"LM Studio API raw response: {response_text}")

                    if response.status != 200:
                        logger.error(f"LM Studio API error: {response.status} - {response_text}")

                        # Try to extract more detailed error information if available
                        try:
                            error_json = json.loads(response_text)
                            if "error" in error_json:
                                logger.error(f"Detailed error: {error_json['error']}")
                        except Exception as parse_error:
                            logger.error(f"Error parsing error response: {str(parse_error)}")

                        return f"Error connecting to the local LLM server: {response.status}"

                    try:
                        result = json.loads(response_text)

                        # Log the full response from LM Studio for debugging
                        logger.debug(f"LM Studio API complete response: {json.dumps(result, indent=2)}")
                    except json.JSONDecodeError as json_error:
                        logger.error(f"Failed to parse response as JSON: {str(json_error)}")
                        return f"The LLM server returned an invalid JSON response: {response_text[:100]}..."
                    if "choices" not in result or not result["choices"]:
                        logger.error(f"No choices in LM Studio response: {result}")
                        return "The LLM server returned an incomplete response."
                    if "message" not in result["choices"][0]:
                        logger.error(f"No message in first choice: {result['choices'][0]}")
                        return "The LLM server returned an invalid response format."
                    content = result["choices"][0]["message"].get("content")
                    if not content:
                        logger.error("No content in message")
                        return "The LLM server returned an empty response."

                    # Log usage data if available
                    token_info = {}
                    if "usage" in result:
                        token_info = {
                            "prompt_tokens": result["usage"].get("prompt_tokens", "N/A"),
                            "completion_tokens": result["usage"].get("completion_tokens", "N/A"),
                            "total_tokens": result["usage"].get("total_tokens", "N/A"),
                        }

                        # Log to CSV
                        self.log_usage_to_csv(
                            model=model or config.lmstudio_model or "local-model",
                            token_usage=token_info,
                            duration=duration,
                            request_type="chat",
                        )

                    logger.info(f"Successfully got response from LM Studio: {content[:100]}...")
                    logger.debug(f"Duration: {duration:.2f}s, Token usage: {token_info}")

                    # Return either raw content or validated JSON
                    return self._process_content(content, return_json) if return_json else content
        except Exception as e:
            logger.error(f"Error in HTTP request: {str(e)}")
            return f"Error connecting to the local LLM server: {str(e)}"

    def _process_content(self, content: str, return_json: bool) -> str:
        """
        Process and validate JSON content from the LLM response.
        Extracts JSON from markdown code blocks if present and validates the JSON format.

        Args:
            content: The raw content from the LLM response
            return_json: Whether JSON validation is required

        Returns:
            The validated and cleaned JSON content as a string, or an error message
        """
        # First, check if the response is wrapped in markdown code blocks
        cleaned_content = content

        # Check for markdown code block format: ```json ... ```
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_block_match = re.search(code_block_pattern, content)
        if code_block_match:
            # Extract just the JSON content without the code block markers
            cleaned_content = code_block_match.group(1).strip()
            logger.info(f"Extracted JSON from markdown code block: {cleaned_content[:100]}...")

        try:
            # Try to parse the content as JSON to validate it
            _ = json.loads(cleaned_content)
            logger.debug("Successfully validated JSON response")
            # Return the cleaned JSON content
            return cleaned_content
        except json.JSONDecodeError:
            # Try one more approach - sometimes there might be extra text before or after
            try:
                # Look for patterns that might be JSON objects
                json_pattern = r"(\{[\s\S]*\})"
                json_match = re.search(json_pattern, cleaned_content)
                if json_match:
                    potential_json = json_match.group(1)
                    # Validate JSON
                    json.loads(potential_json)
                    logger.info("Found and extracted valid JSON object using regex")
                    return potential_json
            except (json.JSONDecodeError, Exception) as nested_error:
                logger.debug(f"Second JSON extraction attempt failed: {str(nested_error)}")
            logger.warning("LM Studio returned non-JSON response when JSON was requested")
            logger.debug(f"Invalid JSON response: {content}")
            return f"Error: The LLM returned invalid JSON format: {content[:100]}..."

    async def chat_with_file(
        self, model: str, system_prompt: str, user_prompt: str, files: list[LLMFile], return_json=False
    ) -> str:
        try:
            start_time = time.time()

            file_contents = []
            for file in files:
                extracted_content = get_file_content_for_filename(file.filename)
                if not extracted_content:
                    extracted_content = extract_text(file)
                    set_file_content_for_filename(file.filename, extracted_content)
                file_contents.append((file.filename, extracted_content))

            # Process files and add to content
            combined_content = ""
            for filename, content in file_contents:
                combined_content += f"\n\nDocument: {filename}\n{content}"

            # Add the file content to the user prompt
            user_prompt += combined_content

            logger.info(f"Sending request with {len(files)} files attached to the prompt")

            result = await self.chat(model, system_prompt, user_prompt, return_json)

            duration = time.time() - start_time
            # Log the full file chat duration (separate from the chat API call itself)
            self.log_usage_to_csv(
                model=model or config.lmstudio_model or "local-model",
                token_usage="See chat logs",  # Token usage already logged in the chat method
                duration=duration,
                request_type="file_chat",
            )

            return result
        except Exception as file_error:
            logger.exception(file_error)
            raise HTTPException(status_code=500, detail=f"Failed to process files: {file_error}") from file_error
