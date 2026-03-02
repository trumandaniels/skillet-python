from __future__ import annotations

import json
import zipfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkilletModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class TargetRuntime(StrEnum):
    CODEX_LIKE = "codex_like"


class LengthProfile(StrEnum):
    COMPACT = "compact"
    MODERATE = "moderate"
    DETAILED = "detailed"


class ActivationPolicy(StrEnum):
    FORCED = "forced"
    AUTONOMOUS = "autonomous"


class EditableSection(StrEnum):
    NAME = "name"
    DESCRIPTION = "description"
    WHEN_TO_USE = "when_to_use"
    BLESSED_PATH = "blessed_path"
    VERIFICATION_CHECKLIST = "verification_checklist"
    DECISION_RULES = "decision_rules"


class OptimizationTarget(StrEnum):
    DELTA = "delta"
    ACTIVATION = "activation"
    BALANCED = "balanced"


class PlanTier(StrEnum):
    FREE = "free"
    BUILDER = "builder"
    TEAM_PILOT = "team_pilot"


class ArtifactFile(SkilletModel):
    path: str
    content: str = ""
    relative_path: str | None = None
    language: str | None = None
    asset_type: str | None = None


class CompiledSkill(SkilletModel):
    skill_id: str
    name: str | None = None
    description: str | None = None
    package_root: str | None = None
    entrypoint: str | None = None
    frontmatter: str | None = None
    skill_md: str = ""
    scripts: list[ArtifactFile] = Field(default_factory=list)
    templates: list[ArtifactFile] = Field(default_factory=list)
    checks: list[ArtifactFile] = Field(default_factory=list)
    references: list[ArtifactFile] = Field(default_factory=list)
    files: list[ArtifactFile] = Field(default_factory=list)


class BundleManifestEntry(SkilletModel):
    skill_id: str
    source_refs: list[str] = Field(default_factory=list)
    suggested_bundle: str | None = None
    command_count: int = 0
    verification_count: int = 0
    task_family: str = ""
    toolchains: list[str] = Field(default_factory=list)
    output_contract: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    merged_skill_ids: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)


class BundleManifest(SkilletModel):
    bundle_name: str
    skill_count: int = 0
    bundle_target: int = Field(ge=1, le=3)
    target_runtime: TargetRuntime = TargetRuntime.CODEX_LIKE
    length_profile: LengthProfile = LengthProfile.MODERATE
    skills: list[BundleManifestEntry] = Field(default_factory=list)


class ComplexityReport(SkilletModel):
    chunk_count: int = 0
    candidate_count: int = 0
    compiled_skill_count: int = 0
    pruned_candidate_count: int = 0
    aggregate_command_count: int = 0
    aggregate_verification_count: int = 0
    aggregate_source_ref_count: int = 0
    aggregate_branch_count: int = 0
    aggregate_token_count: int = 0
    environment_assumption_count: int = 0
    pairwise_conflict_count: int = 0
    complexity_gated_count: int = 0
    bundle_entropy: float = 0.0


class RecommendedRuntimeProfile(SkilletModel):
    target_runtime: TargetRuntime = TargetRuntime.CODEX_LIKE
    length_profile: LengthProfile = LengthProfile.MODERATE
    bundle_size: int = 0
    activation_mode: str = "explicit_activation_terms"
    prefer_script_execution: bool = False
    prefer_checklists: bool = False


class SkillBundle(SkilletModel):
    bundle_name: str | None = None
    files: list[ArtifactFile] = Field(default_factory=list)

    @classmethod
    def from_skill_package(cls, skill_package: SkillPackage) -> SkillBundle:
        bundle_name = skill_package.bundle_name
        if bundle_name is None and skill_package.bundle_manifest is not None:
            bundle_name = skill_package.bundle_manifest.bundle_name
        return cls(bundle_name=bundle_name, files=skill_package.files)

    def save_to(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in self.files:
                archive.writestr(artifact.path, artifact.content)
        return target

    def extract_to(self, path: str | Path) -> Path:
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
    bundle_name: str | None = None
    manifest_path: str | None = None
    manifest: dict[str, Any] | None = None
    files: list[ArtifactFile] = Field(default_factory=list)
    skills: list[CompiledSkill] = Field(default_factory=list)
    bundle_manifest: BundleManifest | None = None
    complexity_report: ComplexityReport | None = None
    risk_flags: list[str] = Field(default_factory=list)
    activation_terms: list[str] = Field(default_factory=list)
    recommended_runtime_profile: RecommendedRuntimeProfile | None = None

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
        return cls.from_api_payload(
            payload["skill_package"],
            bundle_manifest=payload.get("bundle_manifest"),
            complexity_report=payload.get("complexity_report"),
            risk_flags=payload.get("risk_flags"),
            activation_terms=payload.get("activation_terms"),
            recommended_runtime_profile=payload.get("recommended_runtime_profile"),
        )

    def to_api_payload(self) -> dict[str, Any]:
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
        return SkillBundle.from_skill_package(self)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_api_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

class BuildRequest(SkilletModel):
    corpus_text: str = Field(min_length=1)
    overlap_ratio: float = Field(default=0.10, ge=0.0, le=0.50)
    target_runtime: TargetRuntime = TargetRuntime.CODEX_LIKE
    bundle_target: int = Field(default=2, ge=1, le=3)
    length_profile: LengthProfile = LengthProfile.MODERATE
    emit_scripts: bool = True
    emit_checks: bool = True


class EvaluateRequest(SkilletModel):
    skill_package: SkillPackage
    activation_policies: list[ActivationPolicy] = Field(
        default_factory=lambda: [
            ActivationPolicy.FORCED,
            ActivationPolicy.AUTONOMOUS,
        ]
    )
    bundle_ablation: bool = True
    length_ablation: bool = True
    discovery_ablation: bool = False

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

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "skill_package": self.skill_package.to_api_payload(),
            "activation_policies": [policy.value for policy in self.activation_policies],
            "bundle_ablation": self.bundle_ablation,
            "length_ablation": self.length_ablation,
            "discovery_ablation": self.discovery_ablation,
        }


class BenchmarkTask(SkilletModel):
    task_id: str
    prompt: str = Field(min_length=1)
    expected_signals: list[str] = Field(default_factory=list)
    focus_section: EditableSection | None = None


class ProposedSkillEdit(SkilletModel):
    skill_id: str | None = None
    section: EditableSection
    value: Any


class RefinePartitionMetrics(SkilletModel):
    forced_pass_rate: float = 0.0
    autonomous_pass_rate: float = 0.0
    activation_rate: float = 0.0
    regression_count: int = 0
    token_estimate: int = 0


class RefinePartitionDelta(SkilletModel):
    forced_pass_rate: float = 0.0
    autonomous_pass_rate: float = 0.0
    activation_rate: float = 0.0
    regression_count: int = 0
    token_estimate: int = 0


class RefinePartitionComparison(SkilletModel):
    before_metrics: RefinePartitionMetrics
    after_metrics: RefinePartitionMetrics
    delta_metrics: RefinePartitionDelta


class AcceptedRejectedEdit(SkilletModel):
    skill_id: str
    section: EditableSection
    status: str
    reason: str
    before_value: str | list[str] | None = None
    after_value: str | list[str] | None = None


class RefineComplexityDelta(SkilletModel):
    before_token_estimate: int = 0
    after_token_estimate: int = 0
    delta_tokens: int = 0
    before_editable_item_count: int = 0
    after_editable_item_count: int = 0
    delta_editable_items: int = 0


class HoldoutSafetyResult(RefinePartitionComparison):
    accepted: bool
    within_tolerance: bool


class RefineRequest(SkilletModel):
    skill_package: SkillPackage
    proposed_edits: list[ProposedSkillEdit] = Field(default_factory=list)
    dev_tasks: list[BenchmarkTask] = Field(min_length=1)
    holdout_tasks: list[BenchmarkTask] = Field(min_length=1)
    edit_budget: int = Field(ge=1)
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED

    def to_api_payload(self) -> dict[str, Any]:
        return {
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


class EvaluationReport(SkilletModel):
    conditions: list[str] = Field(default_factory=list)
    harness: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    ablations: dict[str, Any] = Field(default_factory=dict)


class RefineResult(SkilletModel):
    refined_skill_package: SkillPackage
    accepted_edits: list[AcceptedRejectedEdit] = Field(default_factory=list)
    rejected_edits: list[AcceptedRejectedEdit] = Field(default_factory=list)
    complexity_delta: RefineComplexityDelta
    dev_revalidation: RefinePartitionComparison
    holdout_safety_result: HoldoutSafetyResult

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> RefineResult:
        data = dict(payload)
        data["refined_skill_package"] = SkillPackage.from_api_payload(
            payload["refined_skill_package"]
        )
        return cls.model_validate(data)


class TokenUsageSummary(SkilletModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ApiKeySummary(SkilletModel):
    id: str
    label: str
    key_prefix: str
    created_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None
    current_month_runs: dict[str, int] = Field(default_factory=dict)
    current_month_tokens: TokenUsageSummary = Field(default_factory=TokenUsageSummary)
    providers_used: list[str] = Field(default_factory=list)


class IssuedApiKey(SkilletModel):
    key: ApiKeySummary
    secret: str


class UsageByApiKey(SkilletModel):
    key_id: str
    label: str
    key_prefix: str
    revoked_at: datetime | None
    runs: dict[str, int] = Field(default_factory=dict)
    tokens: TokenUsageSummary
    providers_used: list[str] = Field(default_factory=list)


class UsageSummary(SkilletModel):
    month: str
    plan: PlanTier
    usage: dict[str, int] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=dict)
    remaining: dict[str, int] = Field(default_factory=dict)
    usage_percent: dict[str, int] = Field(default_factory=dict)
    warning_thresholds_triggered: list[int] = Field(default_factory=list)
    by_api_key: list[UsageByApiKey] = Field(default_factory=list)
    token_totals: TokenUsageSummary
    billing_period_end: datetime | None


class SkillDraft(SkilletModel):
    corpus_text: str = Field(min_length=1)
    overlap_ratio: float = Field(default=0.10, ge=0.0, le=0.50)
    target_runtime: TargetRuntime = TargetRuntime.CODEX_LIKE
    bundle_target: int = Field(default=2, ge=1, le=3)
    length_profile: LengthProfile = LengthProfile.MODERATE
    emit_scripts: bool = True
    emit_checks: bool = True

    @classmethod
    def from_corpus(cls, corpus_text: str, **kwargs: Any) -> SkillDraft:
        return cls(corpus_text=corpus_text, **kwargs)

    def to_request(self) -> BuildRequest:
        return BuildRequest(**self.model_dump(mode="python"))


class SkillDocument(SkilletModel):
    text: str = Field(min_length=1)
    title: str | None = None
