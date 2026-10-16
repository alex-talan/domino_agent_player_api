# Prompt Providers

Reusable LangChain chat prompt providers.

Import these providers through `prompt_providers` (for example,
`from prompt_providers import StaticPromptProvider`). Internal imports are
package-qualified so this module can be used alongside AI-Core's agents and
other packages without top-level `domain` or `infrastructure` collisions.

## API

`PromptProvider[ContextT, PromptT]` keeps the domain contract framework-neutral.
The included adapters accept an application-defined context and return a
LangChain `ChatPromptTemplate` asynchronously:

- `StaticPromptProvider` for fixed system prompts.
- `Jinja2PromptProvider` for trusted inline Jinja2 templates.
- `LangSmithPromptProvider` for trusted prompts managed in LangSmith.
- `MarkdownFilePromptProvider` for trusted Jinja2 templates stored as Markdown.

```python
provider = Jinja2PromptProvider("Answer in {{ language }}.")
prompt = await provider.get_prompt({"language": "English"})
messages = prompt.invoke({}).messages
```

```python
import os

provider = LangSmithPromptProvider(
	"support:production",
	langsmith_api_key=os.environ["LANGSMITH_API_KEY"],
)
prompt = await provider.get_prompt({"language": "English"})
await provider.close()
```

Only use Jinja2 templates from trusted sources. Pass user input as template
variables, never as template source.

LangSmith prompts are executable configuration. This provider disables model
loading, environment-secret resolution, and public prompt pulls.