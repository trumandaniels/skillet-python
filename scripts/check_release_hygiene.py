#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def main() -> int:
    errors: list[str] = []

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    urls = project.get("urls", {})

    name = project.get("name")
    version = project.get("version")

    if name != "skillet-sdk":
        errors.append(f"project.name must be 'skillet-sdk' (found: {name!r})")

    if not isinstance(version, str) or not SEMVER_RE.fullmatch(version):
        errors.append(f"project.version must be semantic versioning (found: {version!r})")

    repository_url = urls.get("Repository")
    if not isinstance(repository_url, str) or not repository_url.startswith("https://"):
        errors.append(
            f"project.urls.Repository must be an https URL (found: {repository_url!r})"
        )
    elif "YOUR_ORG" in repository_url or "example.com" in repository_url:
        errors.append(
            f"project.urls.Repository contains placeholder content: {repository_url!r}"
        )

    for key, value in urls.items():
        if not isinstance(value, str) or not value.startswith("https://"):
            errors.append(f"project.urls.{key} must be an https URL (found: {value!r})")
            continue
        if "YOUR_ORG" in value or "example.com" in value:
            errors.append(f"project.urls.{key} contains placeholder content: {value!r}")

    readme_text = README.read_text(encoding="utf-8")
    if "pip install skillet-sdk" not in readme_text:
        errors.append("README must include canonical install command: 'pip install skillet-sdk'")

    ref_type = os.environ.get("GITHUB_REF_TYPE")
    ref_name = os.environ.get("GITHUB_REF_NAME")
    if ref_type == "tag" and isinstance(version, str):
        allowed_tags = {version, f"v{version}"}
        if ref_name not in allowed_tags:
            errors.append(
                "Tagged releases must match pyproject version. "
                f"Expected one of {sorted(allowed_tags)}, found {ref_name!r}."
            )

    if errors:
        print("Release hygiene checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Release hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
