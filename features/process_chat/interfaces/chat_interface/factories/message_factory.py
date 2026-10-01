from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

class LangchainMessageFactory:
    @staticmethod
    def create_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Creates Langchain message objects from a list of dictionaries.

        Args:
            messages: A list of dictionaries, where each dictionary
                      should have 'role' and 'content' keys.

        Returns:
            A list of Langchain BaseMessage objects (HumanMessage,
            SystemMessage, or AIMessage).
        """
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not role or not content:
                print(f"Warning: Skipping message due to missing role or content: {msg}")
                continue

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                print(f"Warning: Unsupported role '{role}' encountered. Treating as user message.")
                lc_messages.append(HumanMessage(content=content))

        return lc_messages
