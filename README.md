# Skillet Python SDK

Python client for the Skillet API.

## Install

```bash
pip install skillet
```

## Quickstart

```python
from skillet import Client

client = Client(api_key="sk-...")
package = client.build("Your corpus text here...")

# Export to disk
package.bundle().extract_to("./my-skill")

# Evaluate
report = client.evaluate(package)

# Refine
result = client.refine(package, dev_tasks=[...], holdout_tasks=[...], edit_budget=3)
```

## Models And Provider Keys

You can set a default model and provider keys once on the client:

```python
from skillet import Client

client = Client(
    api_key="sk-...",
    model="gpt-5-nano",
    model_provider_keys={"openai": "sk-openai-key"},
)

package = client.build("Your corpus text here...")
report = client.evaluate(package)
result = client.refine(
    package,
    dev_tasks=[...],
    holdout_tasks=[...],
    edit_budget=3,
)
```

`build` and `refine` use those settings for provider-backed structured extraction and refinement proposal generation. `evaluate` can also use them for provider-backed evaluation runs.

Or override them per request:

```python
package = client.build(
    "Your corpus text here...",
    model="gemini-3-flash",
    model_provider_keys={"gemini": "your-google-ai-studio-key"},
)

report = client.evaluate(
    package,
    model="gpt-5-nano",
    model_provider_keys={"openai": "sk-openai-key"},
)

result = client.refine(
    package,
    dev_tasks=[...],
    holdout_tasks=[...],
    edit_budget=3,
    model="gemini-3-flash",
    model_provider_keys={"gemini": "your-google-ai-studio-key"},
)
```

The same fields are also available on `BuildRequest`, `EvaluateRequest`, and `RefineRequest`.
