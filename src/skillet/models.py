from __future__ import annotations

import json
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ALLOWED_PROVIDER_KEYS = frozenset({
    "openai", "anthropic", "google", "gemini", "azure", "bedrock",
})
_MAX_PROVIDER_KEY_LENGTH = 256
_MAX_PROVIDER_KEY_COUNT = 5


def _validate_model_provider_keys(
    keys: dict[str, str] | None,
) -> dict[str, str] | None:
    if keys is None:
        return None
    if len(keys) > _MAX_PROVIDER_KEY_COUNT:
        raise ValueError(
            f"model_provider_keys may contain at most {_MAX_PROVIDER_KEY_COUNT} entries"
        )
    for name, value in keys.items():
        if name not in _ALLOWED_PROVIDER_KEYS:
            raise ValueError(
                f"Unknown provider key '{name}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_PROVIDER_KEYS))}"
            )
        if len(value) > _MAX_PROVIDER_KEY_LENGTH:
            raise ValueError(
                f"Provider key value for '{name}' exceeds {_MAX_PROVIDER_KEY_LENGTH} characters"
            )
    return keys


class SkilletModel(BaseModel):
    """Base model for all Skillet data objects. Permits extra fields for forward compatibility."""

    model_config = ConfigDict(extra="allow")


class TargetRuntime(StrEnum):
    """Runtime environment the compiled skill package is optimized for."""

    CODEX_LIKE = "codex_like"


class LengthProfile(StrEnum):
    """Controls the verbosity of generated skill documents."""

    COMPACT = "compact"
    MODERATE = "moderate"
    DETAILED = "detailed"


class ActivationPolicy(StrEnum):
    """How the skill is presented to the agent during evaluation."""

    FORCED = "forced"
    AUTONOMOUS = "autonomous"


class EditableSection(StrEnum):
    """Sections of a compiled skill that can be refined."""

    NAME = "name"
    DESCRIPTION = "description"
    WHEN_TO_USE = "when_to_use"
    BLESSED_PATH = "blessed_path"
    VERIFICATION_CHECKLIST = "verification_checklist"
    DECISION_RULES = "decision_rules"


class OptimizationTarget(StrEnum):
    """Objective function used during skill refinement."""

    DELTA = "delta"
    ACTIVATION = "activation"
    BALANCED = "balanced"


class PlanTier(StrEnum):
    """Subscription plan tiers."""

    FREE = "free"
    BUILDER = "builder"
    TEAM_PILOT = "team_pilot"


class RunProfile(StrEnum):
    """Evaluation run profile controlling cost/coverage trade-offs."""

    INTERACTIVE = "interactive"
    PR = "pr"
    NIGHTLY = "nightly"


class CompareToMode(StrEnum):
    """Baseline comparison mode for evaluation runs."""

    NONE = "none"
    LATEST_MAIN = "latest_main"
    EXPLICIT_BASELINE = "explicit_baseline"


class ArtifactFile(SkilletModel):
    """A single file produced during skill compilation."""

    path: str = Field(description="Relative path of the file within the skill bundle.")
    content: str = Field(default="", description="UTF-8 text content of the file.")
    relative_path: str | None = Field(default=None, description="Alternative relative path used for display purposes.")
    language: str | None = Field(default=None, description="Programming language of the file content, if applicable.")
    asset_type: str | None = Field(default=None, description="Semantic type of the artifact, e.g. 'script', 'template', 'check'.")


class CompiledSkill(SkilletModel):
    """A single compiled skill within a skill package."""

    skill_id: str = Field(description="Unique identifier for this skill.")
    name: str | None = Field(default=None, description="Human-readable skill name.")
    description: str | None = Field(default=None, description="Short description of what the skill teaches the agent.")
    package_root: str | None = Field(default=None, description="Root directory path for this skill's files.")
    entrypoint: str | None = Field(default=None, description="Path to the primary skill document (typically skill.md).")
    frontmatter: str | None = Field(default=None, description="YAML frontmatter block for the skill document.")
    skill_md: str = Field(default="", description="Full Markdown content of the compiled skill document.")
    scripts: list[ArtifactFile] = Field(default_factory=list, description="Executable scripts included in the skill.")
    templates: list[ArtifactFile] = Field(default_factory=list, description="Template files for the skill.")
    checks: list[ArtifactFile] = Field(default_factory=list, description="Verification check files.")
    references: list[ArtifactFile] = Field(default_factory=list, description="Reference material files.")
    files: list[ArtifactFile] = Field(default_factory=list, description="All artifact files belonging to this skill.")


class BundleManifestEntry(SkilletModel):
    """Manifest metadata for a single skill within a bundle."""

    skill_id: str = Field(description="Unique identifier of the skill.")
    source_refs: list[str] = Field(default_factory=list, description="Corpus chunk references that contributed to this skill.")
    suggested_bundle: str | None = Field(default=None, description="Suggested bundle grouping name.")
    command_count: int = Field(default=0, description="Number of actionable commands in the skill.")
    verification_count: int = Field(default=0, description="Number of verification steps in the skill.")
    task_family: str = Field(default="", description="Semantic task family the skill belongs to.")
    toolchains: list[str] = Field(default_factory=list, description="Toolchains referenced by the skill.")
    output_contract: list[str] = Field(default_factory=list, description="Expected output artifacts the skill produces.")
    risk_flags: list[str] = Field(default_factory=list, description="Risk indicators identified during compilation.")
    merged_skill_ids: list[str] = Field(default_factory=list, description="IDs of skills that were merged into this one.")
    asset_types: list[str] = Field(default_factory=list, description="Types of assets included in this skill.")


class BundleManifest(SkilletModel):
    """Top-level manifest describing a compiled skill bundle."""

    bundle_name: str = Field(description="Name of the skill bundle.")
    skill_count: int = Field(default=0, description="Number of skills in the bundle.")
    bundle_target: int = Field(ge=1, le=3, description="Desired maximum number of skills in the bundle.")
    target_runtime: TargetRuntime = Field(default=TargetRuntime.CODEX_LIKE, description="Runtime environment the bundle targets.")
    length_profile: LengthProfile = Field(default=LengthProfile.MODERATE, description="Verbosity level of the generated skill documents.")
    skills: list[BundleManifestEntry] = Field(default_factory=list, description="Per-skill manifest entries.")


class ComplexityReport(SkilletModel):
    """Quantitative complexity metrics from the build pipeline."""

    chunk_count: int = Field(default=0, description="Number of corpus chunks produced during ingestion.")
    candidate_count: int = Field(default=0, description="Number of skill candidates extracted from chunks.")
    compiled_skill_count: int = Field(default=0, description="Number of skills that survived compilation.")
    pruned_candidate_count: int = Field(default=0, description="Number of candidates pruned during consolidation.")
    aggregate_command_count: int = Field(default=0, description="Total actionable commands across all skills.")
    aggregate_verification_count: int = Field(default=0, description="Total verification steps across all skills.")
    aggregate_source_ref_count: int = Field(default=0, description="Total corpus references across all skills.")
    aggregate_branch_count: int = Field(default=0, description="Total decision branches across all skills.")
    aggregate_token_count: int = Field(default=0, description="Estimated token footprint of the entire bundle.")
    environment_assumption_count: int = Field(default=0, description="Number of environment assumptions detected.")
    pairwise_conflict_count: int = Field(default=0, description="Number of pairwise conflicts between skills.")
    complexity_gated_count: int = Field(default=0, description="Number of skills flagged as complexity-gated.")
    bundle_entropy: float = Field(default=0.0, description="Shannon entropy of skill topic distribution within the bundle.")


class RecommendedRuntimeProfile(SkilletModel):
    """Suggested runtime configuration for deploying the skill bundle."""

    target_runtime: TargetRuntime = Field(default=TargetRuntime.CODEX_LIKE, description="Recommended runtime environment.")
    length_profile: LengthProfile = Field(default=LengthProfile.MODERATE, description="Recommended document verbosity.")
    bundle_size: int = Field(default=0, description="Recommended number of skills to load simultaneously.")
    activation_mode: str = Field(default="explicit_activation_terms", description="How the agent should discover and activate skills.")
    prefer_script_execution: bool = Field(default=False, description="Whether the agent should prefer executing scripts over inline instructions.")
    prefer_checklists: bool = Field(default=False, description="Whether the agent should prefer checklist-style verification.")


class SkillBundle(SkilletModel):
    """Filesystem-oriented export wrapper built from a ``SkillPackage``.

    Obtain a bundle via ``SkillPackage.bundle()``.  Use ``extract_to`` to write
    the bundle files into a directory, or ``save_to`` to write a zip archive.
    """

    bundle_name: str | None = Field(default=None, description="Name of the bundle, derived from the skill package manifest.")
    files: list[ArtifactFile] = Field(default_factory=list, description="All artifact files included in the bundle.")

    @classmethod
    def from_skill_package(cls, skill_package: SkillPackage) -> SkillBundle:
        bundle_name = skill_package.bundle_name
        if bundle_name is None and skill_package.bundle_manifest is not None:
            bundle_name = skill_package.bundle_manifest.bundle_name
        return cls(bundle_name=bundle_name, files=skill_package.files)

    def save_to(self, path: str | Path) -> Path:
        """Write the bundle as a zip archive.

        Args:
            path: Destination path for the zip file.  Parent directories are
                created automatically.

        Returns:
            The resolved ``Path`` of the written archive.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in self.files:
                archive.writestr(artifact.path, artifact.content)
        return target

    def extract_to(self, path: str | Path) -> Path:
        """Write the bundle files into a directory.

        Each ``ArtifactFile.path`` is written relative to ``path``.
        Existing files are overwritten.

        Args:
            path: Target directory.  Created if it does not exist.

        Returns:
            The resolved ``Path`` of the target directory.
        """
        target = Path(path)
        if target.exists() and not target.is_dir():
            raise ValueError(f"Export target must be a directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
        for artifact in self.files:
            destination = target / artifact.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(artifact.content, encoding="utf-8")
        return target


class SkillPackage(SkilletModel):
    """Primary response object returned by ``client.build()``.

    Contains the compiled skills, bundle manifest, complexity report, risk flags,
    activation terms, and recommended runtime profile.

    Use ``bundle()`` to get a ``SkillBundle`` for filesystem export, or
    ``write_json(path)`` to persist the raw payload for later reuse.

    Example:
        ```python
        package = client.build("Your corpus text here...")

        # Write raw payload to disk
        package.write_json("./skill-package.json")

        # Export bundle files
        bundle = package.bundle()
        bundle.extract_to("./my-skill")    # as directory
        bundle.save_to("./my-skill.zip")   # as zip archive
        ```
    """

    bundle_name: str | None = Field(default=None, description="Name of the skill bundle.")
    manifest_path: str | None = Field(default=None, description="Path to the bundle manifest file within the package.")
    manifest: dict[str, Any] | None = Field(default=None, description="Raw manifest dictionary, if available.")
    files: list[ArtifactFile] = Field(default_factory=list, description="All artifact files in the package.")
    skills: list[CompiledSkill] = Field(default_factory=list, description="Compiled skill objects in the package.")
    bundle_manifest: BundleManifest | None = Field(default=None, description="Structured bundle manifest with per-skill metadata.")
    complexity_report: ComplexityReport | None = Field(default=None, description="Quantitative complexity metrics from the build pipeline.")
    risk_flags: list[str] = Field(default_factory=list, description="Risk indicators identified during compilation.")
    activation_terms: list[str] = Field(default_factory=list, description="Terms the agent uses to discover and activate skills.")
    recommended_runtime_profile: RecommendedRuntimeProfile | None = Field(default=None, description="Suggested runtime configuration for deploying the bundle.")

    @classmethod
    def from_api_payload(
        cls,
        payload: dict[str, Any],
        *,
        bundle_manifest: BundleManifest | dict[str, Any] | None = None,
        complexity_report: ComplexityReport | dict[str, Any] | None = None,
        risk_flags: list[str] | None = None,
        activation_terms: list[str] | None = None,
        recommended_runtime_profile: RecommendedRuntimeProfile | dict[str, Any] | None = None,
    ) -> SkillPackage:
        """Construct a ``SkillPackage`` from a raw API response dictionary.

        Merges optional top-level metadata fields into the payload before
        validation so callers can pass them separately.

        Args:
            payload: Raw skill-package dictionary from the API.
            bundle_manifest: Optional bundle manifest to merge.
            complexity_report: Optional complexity report to merge.
            risk_flags: Optional risk flags to merge.
            activation_terms: Optional activation terms to merge.
            recommended_runtime_profile: Optional runtime profile to merge.

        Returns:
            A validated ``SkillPackage`` instance.
        """
        data = dict(payload)
        if bundle_manifest is not None:
            data["bundle_manifest"] = bundle_manifest
        if complexity_report is not None:
            data["complexity_report"] = complexity_report
        if risk_flags is not None:
            data["risk_flags"] = risk_flags
        if activation_terms is not None:
            data["activation_terms"] = activation_terms
        if recommended_runtime_profile is not None:
            data["recommended_runtime_profile"] = recommended_runtime_profile
        return cls.model_validate(data)

    @classmethod
    def from_build_response(cls, payload: dict[str, Any]) -> SkillPackage:
        """Construct a ``SkillPackage`` from the full ``/build`` endpoint response.

        Extracts the nested ``skill_package`` key and top-level metadata fields
        automatically.

        Args:
            payload: Full JSON response from the ``/build`` endpoint.

        Returns:
            A validated ``SkillPackage`` instance.
        """
        return cls.from_api_payload(
            payload["skill_package"],
            bundle_manifest=payload.get("bundle_manifest"),
            complexity_report=payload.get("complexity_report"),
            risk_flags=payload.get("risk_flags"),
            activation_terms=payload.get("activation_terms"),
            recommended_runtime_profile=payload.get("recommended_runtime_profile"),
        )

    def to_api_payload(self) -> dict[str, Any]:
        """Serialize the package to a dictionary suitable for API requests.

        Excludes build-time metadata (manifest, complexity report, etc.) that
        is not accepted by downstream endpoints like ``/evaluate`` or ``/refine``.

        Returns:
            A JSON-serializable dictionary.
        """
        return self.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "bundle_manifest",
                "complexity_report",
                "risk_flags",
                "activation_terms",
                "recommended_runtime_profile",
            },
        )

    def bundle(self) -> SkillBundle:
        """Create a ``SkillBundle`` for filesystem export.

        Returns:
            A ``SkillBundle`` containing all artifact files from this package.
        """
        return SkillBundle.from_skill_package(self)

    def write_json(self, path: str | Path) -> Path:
        """Write the API payload to a JSON file on disk.

        Args:
            path: Destination file path. Parent directories are created automatically.

        Returns:
            The resolved ``Path`` of the written file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_api_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

class BuildRequest(SkilletModel):
    """Full build payload for ``client.build(...)``.

    Use this when you want explicit validation before making an HTTP call or
    when you want to build the payload separately from execution.  Pass a
    ``BuildRequest`` directly to ``client.build()`` instead of a string.
    """

    corpus_text: str = Field(
        min_length=1,
        description=(
            "Raw domain corpus (plaintext, markdown, transcript, etc.). "
            "High procedural density leads to high-fidelity skills."
        ),
    )
    overlap_ratio: float = Field(
        default=0.10, ge=0.0, le=0.50,
        description="Fractional chunk overlap (0.0–0.50). Higher values capture cross-chunk dependencies at the cost of redundancy.",
    )
    target_runtime: TargetRuntime = Field(
        default=TargetRuntime.CODEX_LIKE,
        description="Runtime environment the compiled package targets.",
    )
    bundle_target: int = Field(
        default=2, ge=1, le=3,
        description="Desired maximum number of skills in the result bundle.",
    )
    length_profile: LengthProfile = Field(
        default=LengthProfile.MODERATE,
        description="Verbosity level of the generated skill documents.",
    )
    emit_scripts: bool = Field(
        default=True,
        description="Whether to generate executable script artifacts.",
    )
    emit_checks: bool = Field(
        default=True,
        description="Whether to generate verification check artifacts.",
    )
    model_provider_keys: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional LLM provider API keys for the model calls Skillet orchestrates. "
            "Keys are pass-through only: held in memory for this request, never stored or logged. "
            "Accepted key names: `openai`, `gemini`, `anthropic`, etc."
        ),
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional model to use for LLM calls during this operation "
            "(e.g. `gpt-5-nano`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`). "
            "The provider is inferred from the model name and the corresponding key "
            "must be present in `model_provider_keys`. "
            "If omitted, defaults to a Skillet-selected model."
        ),
    )

    @field_validator("model_provider_keys")
    @classmethod
    def validate_provider_keys(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _validate_model_provider_keys(v)


class EvaluateRequest(SkilletModel):
    """Full evaluation payload for ``client.evaluate(...)``.

    Use this when you want explicit validation before making an HTTP call.
    """

    skill_package: SkillPackage = Field(
        description="The skill package to evaluate, as returned by the build endpoint.",
    )
    activation_policies: list[ActivationPolicy] = Field(
        default_factory=lambda: [
            ActivationPolicy.FORCED,
            ActivationPolicy.AUTONOMOUS,
        ],
        description="Policies to test: 'forced' injects the skill directly; 'autonomous' requires the agent to discover it.",
    )
    bundle_ablation: bool = Field(
        default=True,
        description="Run tests with and without other skills in the bundle to measure interference.",
    )
    length_ablation: bool = Field(
        default=True,
        description="Run tests across different length profiles to measure sensitivity.",
    )
    discovery_ablation: bool = Field(
        default=False,
        description="Run tests with varying discovery hints to measure activation robustness.",
    )
    suite: dict[str, Any] | None = Field(
        default=None,
        description="Optional suite definition or suite reference payload.",
    )
    run_profile: RunProfile = Field(
        default=RunProfile.INTERACTIVE,
        description="Execution profile controlling cost and coverage.",
    )
    compare_to: CompareToMode = Field(
        default=CompareToMode.NONE,
        description="Baseline comparison mode for the evaluation run.",
    )
    baseline_eval_id: str | None = Field(
        default=None,
        description="Required when `compare_to='explicit_baseline'`.",
    )
    quality_gate: dict[str, Any] | None = Field(
        default=None,
        description="Optional quality gate thresholds for regressions and activation.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional hard cap on total estimated tokens consumed by the evaluation run.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=1,
        description="Optional hard cap on total tool invocations across the run.",
    )
    max_wall_time_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Optional hard cap on wall-clock runtime for the full evaluation run.",
    )
    max_estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional conservative upper bound on estimated run cost in USD.",
    )
    model_provider_keys: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional LLM provider API keys for the model calls Skillet orchestrates. "
            "Keys are pass-through only: held in memory for this request, never stored or logged. "
            "Accepted key names: `openai`, `gemini`, `anthropic`, etc."
        ),
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional model to use for LLM calls during evaluation runs "
            "(e.g. `gpt-5-nano`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`). "
            "The provider is inferred from the model name and the corresponding key "
            "must be present in `model_provider_keys`. "
            "If omitted, defaults to a Skillet-selected model."
        ),
    )

    @field_validator("activation_policies")
    @classmethod
    def validate_activation_policies(
        cls,
        activation_policies: list[ActivationPolicy],
    ) -> list[ActivationPolicy]:
        deduped = list(dict.fromkeys(activation_policies))
        if not deduped:
            raise ValueError("activation_policies must include at least one policy")
        return deduped

    @field_validator("model_provider_keys")
    @classmethod
    def validate_provider_keys(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _validate_model_provider_keys(v)

    @model_validator(mode="after")
    def validate_baseline_fields(self) -> "EvaluateRequest":
        if (
            self.compare_to == CompareToMode.EXPLICIT_BASELINE
            and not self.baseline_eval_id
        ):
            raise ValueError(
                "baseline_eval_id is required when compare_to='explicit_baseline'"
            )
        return self

    def to_api_payload(self) -> dict[str, Any]:
        payload = {
            "skill_package": self.skill_package.to_api_payload(),
            "run_profile": self.run_profile.value,
            "compare_to": self.compare_to.value,
            "activation_policies": [policy.value for policy in self.activation_policies],
            "bundle_ablation": self.bundle_ablation,
            "length_ablation": self.length_ablation,
            "discovery_ablation": self.discovery_ablation,
        }
        if self.suite is not None:
            payload["suite"] = self.suite
        if self.baseline_eval_id is not None:
            payload["baseline_eval_id"] = self.baseline_eval_id
        if self.quality_gate is not None:
            payload["quality_gate"] = self.quality_gate
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.max_tool_calls is not None:
            payload["max_tool_calls"] = self.max_tool_calls
        if self.max_wall_time_seconds is not None:
            payload["max_wall_time_seconds"] = self.max_wall_time_seconds
        if self.max_estimated_cost_usd is not None:
            payload["max_estimated_cost_usd"] = self.max_estimated_cost_usd
        if self.model_provider_keys is not None:
            payload["model_provider_keys"] = self.model_provider_keys
        if self.model is not None:
            payload["model"] = self.model
        return payload


class BenchmarkTask(SkilletModel):
    """A single task used in evaluation or refinement benchmarks."""

    task_id: str = Field(description="Unique identifier for the benchmark task.")
    prompt: str = Field(min_length=1, description="The prompt or instruction given to the agent.")
    expected_signals: list[str] = Field(default_factory=list, description="Signals expected in a correct agent response.")
    focus_section: EditableSection | None = Field(default=None, description="Skill section this task is designed to exercise.")


class ProposedSkillEdit(SkilletModel):
    """A proposed edit to a specific section of a compiled skill."""

    skill_id: str | None = Field(default=None, description="ID of the skill to edit. If null, applies to the first skill.")
    section: EditableSection = Field(description="Which section of the skill to modify.")
    value: Any = Field(description="New value for the section.")


class RefinePartitionMetrics(SkilletModel):
    """Absolute performance metrics for a single evaluation partition."""

    forced_pass_rate: float = Field(default=0.0, description="Pass rate when the skill is force-injected.")
    autonomous_pass_rate: float = Field(default=0.0, description="Pass rate when the agent must discover the skill.")
    activation_rate: float = Field(default=0.0, description="Rate at which the agent activates the skill autonomously.")
    regression_count: int = Field(default=0, description="Number of tasks that regressed compared to baseline.")
    token_estimate: int = Field(default=0, description="Estimated token cost for this partition.")


class RefinePartitionDelta(SkilletModel):
    """Change in metrics between before and after refinement for a partition."""

    forced_pass_rate: float = Field(default=0.0, description="Change in forced pass rate.")
    autonomous_pass_rate: float = Field(default=0.0, description="Change in autonomous pass rate.")
    activation_rate: float = Field(default=0.0, description="Change in activation rate.")
    regression_count: int = Field(default=0, description="Change in regression count.")
    token_estimate: int = Field(default=0, description="Change in estimated token cost.")


class RefinePartitionComparison(SkilletModel):
    """Before/after comparison of partition metrics with computed deltas."""

    before_metrics: RefinePartitionMetrics = Field(description="Metrics before refinement.")
    after_metrics: RefinePartitionMetrics = Field(description="Metrics after refinement.")
    delta_metrics: RefinePartitionDelta = Field(description="Computed difference between before and after.")


class AcceptedRejectedEdit(SkilletModel):
    """Record of a proposed edit that was accepted or rejected during refinement."""

    skill_id: str = Field(description="ID of the skill the edit targeted.")
    section: EditableSection = Field(description="Section of the skill that was edited.")
    status: str = Field(description="Outcome: 'accepted' or 'rejected'.")
    reason: str = Field(description="Explanation for why the edit was accepted or rejected.")
    before_value: str | list[str] | None = Field(default=None, description="Section value before the edit.")
    after_value: str | list[str] | None = Field(default=None, description="Section value after the edit, if accepted.")


class RefineComplexityDelta(SkilletModel):
    """Change in complexity metrics caused by refinement."""

    before_token_estimate: int = Field(default=0, description="Token estimate before refinement.")
    after_token_estimate: int = Field(default=0, description="Token estimate after refinement.")
    delta_tokens: int = Field(default=0, description="Net change in token estimate.")
    before_editable_item_count: int = Field(default=0, description="Number of editable items before refinement.")
    after_editable_item_count: int = Field(default=0, description="Number of editable items after refinement.")
    delta_editable_items: int = Field(default=0, description="Net change in editable item count.")


class HoldoutSafetyResult(RefinePartitionComparison):
    """Holdout partition safety check result."""

    accepted: bool = Field(description="Whether the refinement passed holdout safety checks.")
    within_tolerance: bool = Field(description="Whether metric changes are within acceptable tolerance bounds.")


class RefineRequest(SkilletModel):
    """Full refinement payload for ``client.refine(...)``.

    Use this when you want explicit validation before making an HTTP call.
    """

    skill_package: SkillPackage = Field(
        description="The base skill package to refine, as returned by the build endpoint.",
    )
    proposed_edits: list[ProposedSkillEdit] = Field(
        default_factory=list,
        description="Ordered list of section edits to test against the dev partition.",
    )
    dev_tasks: list[BenchmarkTask] = Field(
        min_length=1,
        description="Benchmark tasks used to score candidate edits during refinement.",
    )
    holdout_tasks: list[BenchmarkTask] = Field(
        min_length=1,
        description="Held-out benchmark tasks used for safety validation after refinement.",
    )
    edit_budget: int = Field(
        ge=1,
        description="Maximum number of edits the refiner may apply.",
    )
    optimization_target: OptimizationTarget = Field(
        default=OptimizationTarget.BALANCED,
        description="Objective function: 'delta' maximizes pass-rate improvement, 'activation' maximizes autonomous discovery, 'balanced' optimizes both.",
    )
    model_provider_keys: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional LLM provider API keys for the model calls Skillet orchestrates. "
            "Keys are pass-through only: held in memory for this request, never stored or logged. "
            "Accepted key names: `openai`, `gemini`, `anthropic`, etc."
        ),
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional model to use for LLM calls during refinement "
            "(e.g. `gpt-5-nano`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`). "
            "The provider is inferred from the model name and the corresponding key "
            "must be present in `model_provider_keys`. "
            "If omitted, defaults to a Skillet-selected model."
        ),
    )

    @field_validator("model_provider_keys")
    @classmethod
    def validate_provider_keys(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        return _validate_model_provider_keys(v)

    def to_api_payload(self) -> dict[str, Any]:
        payload = {
            "skill_package": self.skill_package.to_api_payload(),
            "proposed_edits": [
                edit.model_dump(mode="json", exclude_none=True) for edit in self.proposed_edits
            ],
            "dev_tasks": [
                task.model_dump(mode="json", exclude_none=True) for task in self.dev_tasks
            ],
            "holdout_tasks": [
                task.model_dump(mode="json", exclude_none=True) for task in self.holdout_tasks
            ],
            "edit_budget": self.edit_budget,
            "optimization_target": self.optimization_target.value,
        }
        if self.model_provider_keys is not None:
            payload["model_provider_keys"] = self.model_provider_keys
        if self.model is not None:
            payload["model"] = self.model
        return payload


class EvaluationSuiteRef(SkilletModel):
    """Reference to a repo-managed evaluation suite."""

    suite_id: str | None = None
    suite_version: str | None = None
    path: str | None = None
    description: str | None = None
    git_ref: str | None = None


class EvaluationQualityGate(SkilletModel):
    """Quality gate thresholds for CI gating."""

    max_regressions: int | None = Field(default=None, ge=0)
    min_pass_rate_delta_vs_baseline: float | None = None
    min_activation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_context_footprint_delta_pct: float | None = None


class EvaluationBaseline(SkilletModel):
    """Baseline reference metadata in an evaluation report."""

    mode: CompareToMode = CompareToMode.NONE
    status: str = "not_requested"
    eval_id: str | None = None
    suite_id: str | None = None
    suite_version: str | None = None
    run_profile: RunProfile | None = None
    git_ref: str | None = None


class EvaluationBaselineDeltas(SkilletModel):
    """Metric deltas versus a resolved baseline."""

    available: bool = False
    pass_rate_delta_vs_baseline: float = 0.0
    activation_rate_delta: float = 0.0
    context_footprint_delta_pct: float = 0.0


class EvaluationGateViolation(SkilletModel):
    """A single quality gate threshold violation."""

    code: str
    message: str
    actual: Any | None = None
    threshold: Any | None = None


class EvaluationGateResult(SkilletModel):
    """Pass/fail result of quality gate evaluation."""

    status: str = "not_evaluated"  # "pass" | "fail" | "not_evaluated"
    violations: list[EvaluationGateViolation] = Field(default_factory=list)


class EvaluationTokenUsage(SkilletModel):
    """Token consumption breakdown for an evaluation run."""

    input: int = 0
    output: int = 0
    total: int = 0


class EvaluationCostSummary(SkilletModel):
    """Budget and cost summary for an evaluation run."""

    estimated_cost_usd: float = 0.0
    budget_status: str = "not_requested"  # "ok" | "exceeded" | "not_requested"
    exceeded_dimensions: list[str] = Field(default_factory=list)
    token_usage: EvaluationTokenUsage = Field(default_factory=EvaluationTokenUsage)
    tool_calls: int = 0
    wall_clock_s: float = 0.0


class EvaluationStatusCheckPayload(SkilletModel):
    """CI status check payload for GitHub or similar integrations."""

    name: str
    conclusion: str  # "success" | "failure" | "neutral"
    summary: str
    details_url: str | None = None
    external_id: str | None = None


class EvaluationPullRequestCommentPayload(SkilletModel):
    """PR comment payload for CI integrations."""

    title: str
    body_markdown: str
    sticky_identifier: str
    needs_attention: bool = False


class EvaluationCiSignal(SkilletModel):
    """CI integration signals derived from an evaluation report."""

    status_check: EvaluationStatusCheckPayload
    pull_request_comment: EvaluationPullRequestCommentPayload


class EvaluationReport(SkilletModel):
    """Returned by ``client.evaluate()``.

    Contains evaluation metrics, baseline deltas, gate results, CI signals,
    activation diagnostics, regression summary, and cost information.
    """

    eval_id: str | None = None
    suite_id: str | None = None
    suite_version: str | None = None
    git_ref: str | None = None
    run_profile: RunProfile | None = None
    compare_to: CompareToMode | None = None
    quality_gate: EvaluationQualityGate | None = None
    baseline: EvaluationBaseline | None = None
    baseline_deltas: EvaluationBaselineDeltas | None = None
    gate_result: EvaluationGateResult | None = None
    ci_signal: EvaluationCiSignal | None = None
    conditions: list[str] = Field(default_factory=list, description="Evaluation conditions that were tested.")
    harness: dict[str, Any] = Field(default_factory=dict, description="Benchmark harness configuration used.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Aggregated evaluation metrics.")
    ablations: dict[str, Any] = Field(default_factory=dict, description="Results from ablation experiments.")
    policy_metrics: dict[str, Any] = Field(default_factory=dict, description="Per-policy (forced/autonomous) metrics.")
    marginal_contribution: list[dict[str, Any]] = Field(default_factory=list, description="Per-skill marginal contribution data.")
    activation_diagnostics: dict[str, Any] = Field(default_factory=dict, description="Activation behavior diagnostics.")
    regression_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of regressed tasks.")
    recommended_edits: list[Any] = Field(default_factory=list, description="Suggested edits based on evaluation evidence.")
    cost_summary: EvaluationCostSummary | None = None


class EvaluationJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationJobSubmission(SkilletModel):
    """Returned by ``client.submit_evaluation()`` and ``POST /evaluate``."""

    job_id: str
    status: EvaluationJobStatus = EvaluationJobStatus.PENDING
    poll_url: str | None = None


class EvaluationJob(SkilletModel):
    """Returned by ``client.get_evaluation_job()`` and ``GET /evaluate/{job_id}``."""

    job_id: str
    status: EvaluationJobStatus
    evaluation_report: EvaluationReport | None = None
    error: str | None = None


class RefineResult(SkilletModel):
    """Returned by ``client.refine()``.

    Key fields: ``refined_skill_package``, ``accepted_edits``, ``rejected_edits``,
    ``complexity_delta``, ``dev_revalidation``, and ``holdout_safety_result``.
    """

    refined_skill_package: SkillPackage = Field(description="The skill package with accepted edits applied.")
    accepted_edits: list[AcceptedRejectedEdit] = Field(default_factory=list, description="Edits that improved the target metric and were applied.")
    rejected_edits: list[AcceptedRejectedEdit] = Field(default_factory=list, description="Edits that were tested but did not improve the target metric.")
    complexity_delta: RefineComplexityDelta = Field(description="Change in complexity metrics caused by the refinement.")
    dev_revalidation: RefinePartitionComparison = Field(description="Before/after comparison on the dev partition.")
    holdout_safety_result: HoldoutSafetyResult = Field(description="Safety validation results on the holdout partition.")

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> RefineResult:
        data = dict(payload)
        data["refined_skill_package"] = SkillPackage.from_api_payload(
            payload["refined_skill_package"]
        )
        return cls.model_validate(data)


class TokenUsageSummary(SkilletModel):
    """Aggregated token consumption for a billing period or API key."""

    input_tokens: int = Field(default=0, description="Total input tokens consumed.")
    output_tokens: int = Field(default=0, description="Total output tokens generated.")
    total_tokens: int = Field(default=0, description="Sum of input and output tokens.")


class ApiKeySummary(SkilletModel):
    """Summary of an API key with usage metrics."""

    id: str = Field(description="Unique identifier for the API key.")
    label: str = Field(description="Human-readable label assigned to the key.")
    key_prefix: str = Field(description="First characters of the key for identification.")
    created_at: datetime = Field(description="Timestamp when the key was created.")
    revoked_at: datetime | None = Field(description="Timestamp when the key was revoked, or null if active.")
    last_used_at: datetime | None = Field(description="Timestamp of the most recent API call with this key.")
    current_month_runs: dict[str, int] = Field(default_factory=dict, description="Runs per endpoint attributed to this key in the current month.")
    current_month_tokens: TokenUsageSummary = Field(default_factory=TokenUsageSummary, description="Token consumption for this key in the current month.")
    providers_used: list[str] = Field(default_factory=list, description="Model providers invoked through this key.")


class IssuedApiKey(SkilletModel):
    """Returned by ``create_api_key()`` and ``rotate_api_key()``.

    The ``secret`` field contains the raw API key value and is shown **once only**.
    Store it immediately — it cannot be retrieved again.
    """

    key: ApiKeySummary = Field(description="Metadata for the created or rotated key.")
    secret: str = Field(description="Raw API key secret. Shown once only — store it immediately.")


class UsageByApiKey(SkilletModel):
    """Per-key usage attribution within a billing period."""

    key_id: str = Field(description="Unique identifier of the API key.")
    label: str = Field(description="Human-readable label for the key.")
    key_prefix: str = Field(description="First characters of the key for identification.")
    revoked_at: datetime | None = Field(description="Revocation timestamp, or null if the key is active.")
    runs: dict[str, int] = Field(default_factory=dict, description="Run counts per endpoint attributed to this key.")
    tokens: TokenUsageSummary = Field(description="Token consumption attributed to this key.")
    providers_used: list[str] = Field(default_factory=list, description="Model providers invoked through this key.")


class UsageSummary(SkilletModel):
    """Returned by ``client.get_usage()``.

    Key fields: ``month``, ``plan``, ``usage``, ``limits``, ``remaining``,
    ``usage_percent``, ``warning_thresholds_triggered``, ``by_api_key``,
    ``token_totals``, and ``billing_period_end``.
    """

    month: str = Field(description="Billing month in YYYY-MM format.")
    plan: PlanTier = Field(description="Subscription plan tier during this billing month.")
    usage: dict[str, int] = Field(default_factory=dict, description="Total runs consumed per endpoint.")
    limits: dict[str, int] = Field(default_factory=dict, description="Maximum allowed runs per endpoint.")
    remaining: dict[str, int] = Field(default_factory=dict, description="Runs remaining per endpoint.")
    usage_percent: dict[str, int] = Field(default_factory=dict, description="Percentage of monthly quota consumed per endpoint.")
    warning_thresholds_triggered: list[int] = Field(default_factory=list, description="Usage percentage thresholds that have been crossed.")
    by_api_key: list[UsageByApiKey] = Field(default_factory=list, description="Per-key usage attribution breakdown.")
    token_totals: TokenUsageSummary = Field(description="Aggregated token consumption across all keys.")
    billing_period_end: datetime | None = Field(description="End of the current billing period, or null for free-tier accounts.")


class EvaluationHistoryItem(SkilletModel):
    """Summary of a single evaluation run from ``/app/evaluations``."""

    eval_id: str = Field(description="Unique identifier for the evaluation run.")
    suite_id: str = Field(description="Suite identifier evaluated by the run.")
    suite_version: str | None = Field(default=None, description="Version of the suite, when supplied.")
    run_profile: str = Field(description="Execution profile, such as `interactive`, `pr`, or `nightly`.")
    git_ref: str | None = Field(default=None, description="Git reference associated with the run, when supplied.")
    compare_to: str = Field(description="Requested baseline comparison mode.")
    status: str = Field(description="Persisted evaluation status for the run.")
    model: str | None = Field(default=None, description="Model used for the evaluation, when specified.")
    created_at: datetime = Field(description="Timestamp when the evaluation run was recorded.")
    conditions: list[str] = Field(default_factory=list, description="Evaluation conditions executed for the run.")
    pass_rate_delta: float | None = Field(default=None, description="Pass-rate delta achieved by this run.")
    activation_rate: float | None = Field(default=None, description="Activation rate observed for this run.")
    regression_count: int = Field(default=0, description="Count of regressed tasks for the run.")
    context_footprint_delta_pct: float | None = Field(default=None, description="Context-footprint delta percentage versus baseline when available.")
    token_total: int = Field(default=0, description="Estimated total token usage for the run.")
    tool_calls: int = Field(default=0, description="Estimated tool calls during the run.")
    wall_clock_s: float | None = Field(default=None, description="Approximate wall-clock runtime in seconds.")
    baseline: EvaluationBaseline = Field(default_factory=EvaluationBaseline, description="Resolved baseline metadata for the run.")
    baseline_deltas: EvaluationBaselineDeltas = Field(default_factory=EvaluationBaselineDeltas, description="Computed deltas versus the baseline.")
    gate_result: EvaluationGateResult = Field(default_factory=EvaluationGateResult, description="Gate result and violations for the run.")
    cost_summary: EvaluationCostSummary | None = Field(default=None, description="Budget and cost summary for the run, when available.")


class EvaluationHistory(SkilletModel):
    """Returned by ``client.get_evaluations()``."""

    evaluations: list[EvaluationHistoryItem] = Field(default_factory=list, description="Most recent evaluation runs visible to the authenticated organization.")


class SkillDraft(SkilletModel):
    """Local draft model used by ``SkillSession.from_corpus()``.

    Stores corpus text and build options locally before sending them to the API.
    Call ``draft.to_request()`` to convert it into a ``BuildRequest``.
    """

    corpus_text: str = Field(min_length=1, description="Raw domain corpus text to distill into skills.")
    overlap_ratio: float = Field(default=0.10, ge=0.0, le=0.50, description="Fractional chunk overlap (0.0–0.50).")
    target_runtime: TargetRuntime = Field(default=TargetRuntime.CODEX_LIKE, description="Runtime environment the package targets.")
    bundle_target: int = Field(default=2, ge=1, le=3, description="Desired maximum number of skills in the result.")
    length_profile: LengthProfile = Field(default=LengthProfile.MODERATE, description="Verbosity level of generated skill documents.")
    emit_scripts: bool = Field(default=True, description="Whether to generate executable script artifacts.")
    emit_checks: bool = Field(default=True, description="Whether to generate verification check artifacts.")

    @classmethod
    def from_corpus(cls, corpus_text: str, **kwargs: Any) -> SkillDraft:
        return cls(corpus_text=corpus_text, **kwargs)

    def to_request(self) -> BuildRequest:
        return BuildRequest(**self.model_dump(mode="python"))


class SkillDocument(SkilletModel):
    """A plain-text skill document with optional title metadata."""

    text: str = Field(min_length=1, description="Full text content of the skill document.")
    title: str | None = Field(default=None, description="Optional title for the document.")
