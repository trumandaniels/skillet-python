# Skillet Python SDK

Python client for the Skillet API.

## Install

pip install skillet

## Quickstart

from skillet import Client

client = Client(api_key="sk-...")
package = client.build("Your corpus text here...")

# Export to disk
package.bundle().extract_to("./my-skill")

# Evaluate
report = client.evaluate(package)

# Refine
result = client.refine(package, dev_tasks=[...], holdout_tasks=[...], edit_budget=3)
