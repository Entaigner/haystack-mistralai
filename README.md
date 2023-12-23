# haystack-mistralai

A copy of `components/generators/openai.py` and `components/generators/chat/openai.py` adjusted to use the MistralAI API.

---

**Table of Contents**

- [Installation](#installation)
- [Example](#example)
- [License](#license)

## Installation

```console
git clone https://github.com/Entaigner/haystack-mistralai.git
cd haystack-mistralai
python -m pip install -e .
```

## Example

```python
from haystack.dataclasses import ChatMessage
from haystack_mistralai import MistralAIGenerator
from haystack_mistralai import MistralAIChatGenerator

generator = MistralAIGenerator(model_name="mistral-small")
prompt = "Tell me a joke about trees."
response = generator.run(prompt=prompt)
print(response["replies"][0])
print(response["metadata"][0])

# OR

generator = MistralAIChatGenerator(model_name="mistral-small")
prompt = "Tell me a joke about trees."
messages = [
    ChatMessage.from_user(prompt)
]
response = generator.run(messages=messages)
print(response["replies"][0])
print(response["metadata"][0])
```

## License

`haystack-mistralai` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
