from typing import Optional, List, Callable, Dict, Any

from haystack import component, default_to_dict
from haystack.components.generators.utils import serialize_callback_handler
from haystack.dataclasses import StreamingChunk, ChatMessage

from haystack_mistralai.MistralAIChatGenerator import MistralAIChatGenerator


class MistralAIGenerator(MistralAIChatGenerator):

    system_prompt: str | None = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 5,
        timeout: int = 120,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            max_retries=max_retries,
            timeout=timeout,
            generation_kwargs=generation_kwargs,
            streaming_callback=streaming_callback,
        )
        self.system_prompt = system_prompt

        print()

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self,
            prompt: str,
            generation_kwargs: Optional[Dict[str, Any]] = None
            ):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the __init__ method.
        For more details on the parameters supported by the MistralAI API, refer to the
        MistralAI [documentation](https://docs.mistral.ai/api#operation/createChatCompletion)
        or source [sourcecode](https://github.com/mistralai/client-python/blob/main/src/mistralai/client.py)
        somewhere around line 116.
        :return: A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        message = ChatMessage.from_user(prompt)
        if self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        return super().run(messages, generation_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        callback_name = serialize_callback_handler(
            self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model_name=self.model_name,
            streaming_callback=callback_name,
            api_base_url=self.api_base,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt
        )
