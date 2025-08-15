"""
Mock version of the LMStudio class for testing without requiring real dependencies
"""

class MockLMStudio:
    """
    Mock implementation of LMStudio for testing
    """

    async def chat(self, model, system_prompt, user_prompt, agent="mock-lmstudio", return_json=False):
        """Mock implementation of chat"""
        if return_json:
            return '{"result": "success"}'
        return "Mock response from LMStudio"

    def _process_content(self, content, return_json):
        """Mock implementation of _process_content"""
        if "not JSON" in content:
            return "Error: The LLM returned invalid JSON format"

        if '```json' in content:
            # Extract from code block
            import re
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(code_block_pattern, content)
            if match:
                return match.group(1).strip()

        if '{' in content and '}' in content and not content.strip().startswith('{'):
            # Extract with regex
            import re
            json_pattern = r'(\{[\s\S]*\})'
            match = re.search(json_pattern, content)
            if match:
                return match.group(1)

        # Return as is if it looks like JSON
        if content.strip().startswith('{') and content.strip().endswith('}'):
            return content

        return content

    async def chat_with_file(self, model, system_prompt, user_prompt, files, agent, return_json=False):
        """Mock implementation of chat_with_file"""
        file_names = [f.filename for f in files]
        return f"Mock response with files: {', '.join(file_names)}"
