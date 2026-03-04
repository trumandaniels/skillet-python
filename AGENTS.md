## Shell output hygiene

- **Do not** chain `| tail`, `| head`, `| grep` repeatedly while iterating.
- Instead, **run once**, **save output to a temp file**, then inspect/filter the file.

Example:
```bash
cmd ... > /tmp/out.txt
# then iterate quickly:
grep -n "pattern" /tmp/out.txt
tail -n 50 /tmp/out.txt
head -n 50 /tmp/out.txt
````

## Web fetching (avoid browser)

* Prefer **curl/wget to a temp file** over opening Chrome or manual browsing unless there is a good reason (testing UI etc).

Example:

```bash
curl -L "https://example.com" -o /tmp/page.html
grep -n "keyword" /tmp/page.html
```

## In-Place Mutation Only:

Never create new versions of existing files (e.g., no main_v2.py or script_new.js). Modify existing files directly. If a complete rewrite is necessary, overwrite the existing file.

## No Dead Code

When refactoring or updating logic, immediately remove the old, unused code. Do not comment it out for "future reference." Keep the codebase lean.

# Strict Real-Data Policy

DO NOT generate mock data, synthetic databases, or placeholder APIs unless explicitly instructed by the user or as a part of the testing suite, or as a part of testing the output of the function. If a task requires data to proceed, halt execution and explicitly ask the user for the real data source, schema, or connection string.