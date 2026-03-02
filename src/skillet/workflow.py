from __future__ import annotations

from typing import Any

from .exceptions import SkilletError
from .models import EvaluationReport, RefineResult, SkillDraft, SkillPackage


class SkillSession:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.draft: SkillDraft | None = None
        self.skill_package: SkillPackage | None = None
        self.evaluation_report: EvaluationReport | None = None
        self.refine_result: RefineResult | None = None

    def from_corpus(self, corpus_text: str | SkillDraft, **kwargs: Any) -> SkillSession:
        if isinstance(corpus_text, SkillDraft):
            if kwargs:
                raise TypeError("Keyword arguments are not supported when passing a SkillDraft")
            self.draft = corpus_text
        else:
            self.draft = SkillDraft.from_corpus(corpus_text, **kwargs)
        return self

    def build(self, draft: SkillDraft | None = None) -> SkillPackage:
        if draft is not None:
            self.draft = draft
        if self.draft is None:
            raise SkilletError("Call from_corpus() or pass a SkillDraft before build()")
        self.skill_package = self.client.build(self.draft.to_request())
        return self.skill_package

    def evaluate(
        self,
        skill_package: SkillPackage | None = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        package = skill_package or self.skill_package
        if package is None:
            raise SkilletError("A skill package is required before evaluate()")
        self.evaluation_report = self.client.evaluate(package, **kwargs)
        return self.evaluation_report

    def refine(
        self,
        skill_package: SkillPackage | None = None,
        **kwargs: Any,
    ) -> RefineResult:
        package = skill_package or self.skill_package
        if package is None:
            raise SkilletError("A skill package is required before refine()")
        self.refine_result = self.client.refine(package, **kwargs)
        self.skill_package = self.refine_result.refined_skill_package
        return self.refine_result


SkillWorkflow = SkillSession
