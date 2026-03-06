from __future__ import annotations

from typing import Any

from .exceptions import SkilletError
from .models import BuildArtifact, EvaluationReport, RefineResult, SkillDraft, SkillPackage


class SkillSession:
    """Stateful helper that wraps a ``Client`` for corpus-first sessions.

    ``SkillSession`` stores the current draft, built package, evaluation report,
    and refine result between calls so you can write corpus-first code without
    passing objects around manually.  It does not change the server API — every
    method delegates to the underlying ``Client``.

    Args:
        client: An authenticated ``Client`` instance.

    Example:
        ```python
        from skillet import Client, SkillSession

        client = Client(api_key="sk_live_...")
        session = SkillSession(client)

        package = (
            session
            .from_corpus("Your corpus text here...", bundle_target=2)
            .build()
        )
        report = session.evaluate(activation_policies=["forced", "autonomous"])
        ```
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.draft: SkillDraft | None = None
        self.build_artifact: BuildArtifact | None = None
        self.skill_package: SkillPackage | None = None
        self.evaluation_report: EvaluationReport | None = None
        self.refine_result: RefineResult | None = None

    def from_corpus(self, corpus_text: str | SkillDraft, **kwargs: Any) -> SkillSession:
        """Store a draft locally without sending a network request.

        Use this as the entry point for corpus-first sessions.  The draft is
        validated immediately but not sent to the API until ``build()`` is called.

        Args:
            corpus_text: Raw domain corpus as a string, or a pre-built ``SkillDraft``.
            **kwargs: Forwarded to ``SkillDraft`` when ``corpus_text`` is a string.
                Accepted keys: ``overlap_ratio``, ``target_runtime``, ``bundle_target``,
                ``length_profile``, ``emit_scripts``, ``emit_checks``, ``mode``,
                and ``target_outcome``.

        Returns:
            ``self``, so calls can be chained.

        Example:
            ```python
            session = SkillSession(client).from_corpus(
                "Your corpus text here...",
                bundle_target=2,
                length_profile="moderate",
            )
            ```
        """
        if isinstance(corpus_text, SkillDraft):
            if kwargs:
                raise TypeError("Keyword arguments are not supported when passing a SkillDraft")
            self.draft = corpus_text
        else:
            self.draft = SkillDraft.from_corpus(corpus_text, **kwargs)
        return self

    def build(self, draft: SkillDraft | None = None) -> BuildArtifact:
        """Build the stored draft (or a supplied draft) into a ``BuildArtifact``.

        Calls ``POST /build`` via the underlying client and stores the result on
        the session as ``self.build_artifact``. ``self.skill_package`` is only
        populated when the artifact type is a normal `skill`.

        Args:
            draft: Optional ``SkillDraft`` to build instead of the one stored by
                ``from_corpus()``.

        Returns:
            The built ``BuildArtifact``.

        Raises:
            SkilletError: If no draft has been set and none is supplied.

        Example:
            ```python
            artifact = session.build()
            ```
        """
        if draft is not None:
            self.draft = draft
        if self.draft is None:
            raise SkilletError("Call from_corpus() or pass a SkillDraft before build()")
        self.build_artifact = self.client.build(self.draft.to_request())
        self.skill_package = self.build_artifact.skill_package if self.build_artifact.is_skill else None
        return self.build_artifact

    def evaluate(
        self,
        skill_package: SkillPackage | None = None,
        **kwargs: Any,
    ) -> EvaluationReport:
        """Evaluate the stored package (or a supplied package).

        Calls ``POST /evaluate`` and stores the report as ``self.evaluation_report``.

        Args:
            skill_package: Optional package to evaluate instead of the last built one.
            **kwargs: Forwarded to ``Client.evaluate()``.  Accepted keys:
                ``activation_policies``, ``bundle_ablation``, ``length_ablation``,
                ``discovery_ablation``.

        Returns:
            The ``EvaluationReport``.

        Raises:
            SkilletError: If no normal skill package is available and none is supplied.

        Example:
            ```python
            report = session.evaluate(activation_policies=["forced", "autonomous"])
            ```
        """
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
        """Refine the stored package (or a supplied package).

        Calls ``POST /refine`` and, on success, updates ``self.skill_package``
        to the refined package so subsequent ``evaluate()`` calls use it.

        Args:
            skill_package: Optional package to refine instead of the last built one.
            **kwargs: Forwarded to ``Client.refine()``.  Required keys:
                ``proposed_edits``, ``dev_tasks``, ``holdout_tasks``, ``edit_budget``.
                Optional: ``optimization_target``.

        Returns:
            The ``RefineResult``.

        Raises:
            SkilletError: If no normal skill package is available and none is supplied.

        Example:
            ```python
            refined = session.refine(
                proposed_edits=[{"section": "decision_rules", "value": ["Abort weak IV runs when F < 10"]}],
                dev_tasks=[{"task_id": "dev-1", "prompt": "Stop weak IV runs"}],
                holdout_tasks=[{"task_id": "holdout-1", "prompt": "Stop weak IV runs"}],
                edit_budget=1,
            )
            ```
        """
        package = skill_package or self.skill_package
        if package is None:
            raise SkilletError("A skill package is required before refine()")
        self.refine_result = self.client.refine(package, **kwargs)
        self.skill_package = self.refine_result.refined_skill_package
        return self.refine_result
