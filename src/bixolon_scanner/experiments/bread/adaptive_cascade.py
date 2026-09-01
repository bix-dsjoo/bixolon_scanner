from __future__ import annotations

import argparse
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file


class Stage2Verdict(StrEnum):
    KEEP = "KEEP"
    REFINE = "REFINE"
    BACKGROUND = "BACKGROUND"
    PARTIAL = "PARTIAL"
    MERGED = "MERGED"
    UNRESOLVED = "UNRESOLVED"


class CandidateRoute(StrEnum):
    EARLY_LOCK = "EARLY_LOCK"
    BOX_VERIFY = "BOX_VERIFY"
    IDENTITY_VERIFY = "IDENTITY_VERIFY"
    BOX_THEN_IDENTITY_VERIFY = "BOX_THEN_IDENTITY_VERIFY"


class CandidateOutcome(StrEnum):
    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"
    SEGMENT_RECAPTURE = "SEGMENT_RECAPTURE"
    BACKGROUND = "BACKGROUND"
    UNRESOLVED = "UNRESOLVED"


class RankedClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class Stage2Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Stage2Verdict
    confidence: float = Field(ge=0.0, le=1.0)


class Stage3Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_inlier_score: float = Field(ge=0.0, le=1.0)
    support_agreement: float = Field(ge=0.0, le=1.0)
    verifier_agreement: bool
    prediction_set_size: int = Field(ge=1)
    quality_failure: bool = False
    top3: list[RankedClass] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_ranking(self) -> "Stage3Evidence":
        _validate_top3(self.top3)
        return self


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    detector_score: float = Field(ge=0.0, le=1.0)
    localization_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    catalog_inlier_score: float = Field(ge=0.0, le=1.0)
    support_agreement: float = Field(ge=0.0, le=1.0)
    border_contact: bool = False
    overlap_conflict: bool = False
    top3: list[RankedClass] = Field(min_length=3, max_length=3)
    stage2: Stage2Evidence | None = None
    stage3: Stage3Evidence | None = None
    matched_ground_truth_id: str | None = None

    @model_validator(mode="after")
    def validate_ranking(self) -> "CandidateEvidence":
        _validate_top3(self.top3)
        return self


class SceneEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_quality_failure: bool = False
    capacity_saturated: bool = False
    predicted_count: int | None = Field(default=None, ge=0)
    count_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    unexplained_foreground_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_count_pair(self) -> "SceneEvidence":
        if (self.predicted_count is None) != (self.count_confidence is None):
            raise ValueError("predicted_count and count_confidence must be supplied together")
        return self


class GroundTruthObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(min_length=1)
    class_id: str = Field(min_length=1)


class AdaptiveCascadeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str | int
    perceptual_group_id: str = Field(min_length=1)
    scene: SceneEvidence
    candidates: list[CandidateEvidence]
    ground_truth: list[GroundTruthObject]

    @model_validator(mode="after")
    def validate_identifiers(self) -> "AdaptiveCascadeRecord":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique within an image")
        object_ids = [target.object_id for target in self.ground_truth]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("ground-truth object_id values must be unique within an image")
        known = set(object_ids)
        invalid = sorted(
            {
                candidate.matched_ground_truth_id
                for candidate in self.candidates
                if candidate.matched_ground_truth_id is not None
                and candidate.matched_ground_truth_id not in known
            }
        )
        if invalid:
            raise ValueError(f"candidate references unknown ground truth: {invalid}")
        return self


class AdaptiveCascadePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_minimum_score: float = Field(ge=0.0, le=1.0)
    early_detector_score_minimum: float = Field(ge=0.0, le=1.0)
    early_localization_score_minimum: float = Field(ge=0.0, le=1.0)
    early_quality_score_minimum: float = Field(ge=0.0, le=1.0)
    early_catalog_inlier_minimum: float = Field(ge=0.0, le=1.0)
    early_top1_score_minimum: float = Field(ge=0.0, le=1.0)
    early_top1_margin_minimum: float = Field(ge=0.0, le=1.0)
    early_support_agreement_minimum: float = Field(ge=0.0, le=1.0)
    stage2_confidence_minimum: float = Field(ge=0.0, le=1.0)
    stage3_catalog_inlier_minimum: float = Field(ge=0.0, le=1.0)
    stage3_approval_score_minimum: float = Field(ge=0.0, le=1.0)
    stage3_approval_margin_minimum: float = Field(ge=0.0, le=1.0)
    stage3_support_agreement_minimum: float = Field(ge=0.0, le=1.0)
    count_confidence_minimum: float = Field(ge=0.0, le=1.0)
    unexplained_foreground_maximum: float = Field(ge=0.0, le=1.0)
    require_exact_count: bool = True
    require_foreground_evidence: bool = True
    maximum_prediction_set_size: Literal[3] = 3

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "AdaptiveCascadePolicy":
        if self.proposal_minimum_score > self.early_detector_score_minimum:
            raise ValueError("proposal threshold cannot exceed early-lock detector threshold")
        return self


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    route: CandidateRoute
    outcome: CandidateOutcome
    reason: str
    used_stage2: bool
    used_stage3: bool
    predicted_class_id: str | None = None
    top3: list[str] = Field(default_factory=list, max_length=3)
    matched_ground_truth_id: str | None = None


class ImageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str | int
    status: Literal["SEGMENTATION", "IMAGE_RECAPTURE"]
    diagnostic_reasons: list[str]
    candidates: list[CandidateDecision]
    public_candidate_ids: list[str]
    early_lock_count: int
    stage2_call_count: int
    stage3_call_count: int


def _validate_top3(values: list[RankedClass]) -> None:
    scores = [value.score for value in values]
    if scores != sorted(scores, reverse=True):
        raise ValueError("top3 must be sorted by descending score")
    class_ids = [value.class_id for value in values]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("top3 class ids must be unique")


def _margin(values: list[RankedClass]) -> float:
    return values[0].score - values[1].score


def _fast_box_safe(candidate: CandidateEvidence, policy: AdaptiveCascadePolicy) -> bool:
    return (
        candidate.detector_score >= policy.early_detector_score_minimum
        and candidate.localization_score >= policy.early_localization_score_minimum
        and not candidate.border_contact
        and not candidate.overlap_conflict
    )


def _fast_identity_safe(candidate: CandidateEvidence, policy: AdaptiveCascadePolicy) -> bool:
    return (
        candidate.quality_score >= policy.early_quality_score_minimum
        and candidate.catalog_inlier_score >= policy.early_catalog_inlier_minimum
        and candidate.top3[0].score >= policy.early_top1_score_minimum
        and _margin(candidate.top3) >= policy.early_top1_margin_minimum
        and candidate.support_agreement >= policy.early_support_agreement_minimum
    )


def _route(box_safe: bool, identity_safe: bool) -> CandidateRoute:
    if box_safe and identity_safe:
        return CandidateRoute.EARLY_LOCK
    if not box_safe and not identity_safe:
        return CandidateRoute.BOX_THEN_IDENTITY_VERIFY
    if not box_safe:
        return CandidateRoute.BOX_VERIFY
    return CandidateRoute.IDENTITY_VERIFY


def _candidate_decision(
    candidate: CandidateEvidence,
    *,
    route: CandidateRoute,
    outcome: CandidateOutcome,
    reason: str,
    used_stage2: bool,
    used_stage3: bool,
    ranking: list[RankedClass] | None = None,
) -> CandidateDecision:
    values = candidate.top3 if ranking is None else ranking
    predicted = values[0].class_id if outcome is CandidateOutcome.APPROVED else None
    top3 = [value.class_id for value in values] if outcome is CandidateOutcome.UNKNOWN else []
    return CandidateDecision(
        candidate_id=candidate.candidate_id,
        route=route,
        outcome=outcome,
        reason=reason,
        used_stage2=used_stage2,
        used_stage3=used_stage3,
        predicted_class_id=predicted,
        top3=top3,
        matched_ground_truth_id=candidate.matched_ground_truth_id,
    )


def route_candidate(
    candidate: CandidateEvidence,
    policy: AdaptiveCascadePolicy,
) -> CandidateDecision:
    box_safe = _fast_box_safe(candidate, policy)
    identity_safe = _fast_identity_safe(candidate, policy)
    route = _route(box_safe, identity_safe)
    if route is CandidateRoute.EARLY_LOCK:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.APPROVED,
            reason="FAST_EVIDENCE_SAFE",
            used_stage2=False,
            used_stage3=False,
        )

    used_stage2 = not box_safe
    if used_stage2:
        stage2 = candidate.stage2
        if stage2 is None or stage2.confidence < policy.stage2_confidence_minimum:
            return _candidate_decision(
                candidate,
                route=route,
                outcome=CandidateOutcome.UNRESOLVED,
                reason="STAGE2_EVIDENCE_UNAVAILABLE_OR_LOW_CONFIDENCE",
                used_stage2=True,
                used_stage3=False,
            )
        if stage2.verdict is Stage2Verdict.BACKGROUND:
            return _candidate_decision(
                candidate,
                route=route,
                outcome=CandidateOutcome.BACKGROUND,
                reason="STAGE2_BACKGROUND_CONFIRMED",
                used_stage2=True,
                used_stage3=False,
            )
        if stage2.verdict is Stage2Verdict.PARTIAL:
            return _candidate_decision(
                candidate,
                route=route,
                outcome=CandidateOutcome.SEGMENT_RECAPTURE,
                reason="STAGE2_PARTIAL_OBJECT",
                used_stage2=True,
                used_stage3=False,
            )
        if stage2.verdict in {Stage2Verdict.MERGED, Stage2Verdict.UNRESOLVED}:
            return _candidate_decision(
                candidate,
                route=route,
                outcome=CandidateOutcome.UNRESOLVED,
                reason=f"STAGE2_{stage2.verdict.value}",
                used_stage2=True,
                used_stage3=False,
            )

    used_stage3 = not identity_safe
    if not used_stage3:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.APPROVED,
            reason="STAGE2_BOX_VERIFIED_FAST_IDENTITY_SAFE",
            used_stage2=used_stage2,
            used_stage3=False,
        )

    stage3 = candidate.stage3
    if stage3 is None:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.SEGMENT_RECAPTURE,
            reason="STAGE3_EVIDENCE_UNAVAILABLE",
            used_stage2=used_stage2,
            used_stage3=True,
        )
    if stage3.quality_failure:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.SEGMENT_RECAPTURE,
            reason="STAGE3_QUALITY_FAILURE",
            used_stage2=used_stage2,
            used_stage3=True,
            ranking=stage3.top3,
        )
    if stage3.catalog_inlier_score < policy.stage3_catalog_inlier_minimum:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.SEGMENT_RECAPTURE,
            reason="STAGE3_OUT_OF_CATALOG",
            used_stage2=used_stage2,
            used_stage3=True,
            ranking=stage3.top3,
        )
    if stage3.prediction_set_size > policy.maximum_prediction_set_size:
        return _candidate_decision(
            candidate,
            route=route,
            outcome=CandidateOutcome.SEGMENT_RECAPTURE,
            reason="STAGE3_PREDICTION_SET_TOO_LARGE",
            used_stage2=used_stage2,
            used_stage3=True,
            ranking=stage3.top3,
        )
    stage3_approved = (
        stage3.prediction_set_size == 1
        and stage3.verifier_agreement
        and stage3.top3[0].score >= policy.stage3_approval_score_minimum
        and _margin(stage3.top3) >= policy.stage3_approval_margin_minimum
        and stage3.support_agreement >= policy.stage3_support_agreement_minimum
    )
    return _candidate_decision(
        candidate,
        route=route,
        outcome=(CandidateOutcome.APPROVED if stage3_approved else CandidateOutcome.UNKNOWN),
        reason=("STAGE3_SINGLETON_SAFE" if stage3_approved else "STAGE3_TOP3_SAFE"),
        used_stage2=used_stage2,
        used_stage3=True,
        ranking=stage3.top3,
    )


def decide_image(
    record: AdaptiveCascadeRecord,
    policy: AdaptiveCascadePolicy,
) -> ImageDecision:
    early_reasons = []
    if record.scene.hard_quality_failure:
        early_reasons.append("HARD_QUALITY_FAILURE")
    if record.scene.capacity_saturated:
        early_reasons.append("DETECTOR_CAPACITY_SATURATED")
    if early_reasons:
        return ImageDecision(
            image_id=record.image_id,
            status="IMAGE_RECAPTURE",
            diagnostic_reasons=early_reasons,
            candidates=[],
            public_candidate_ids=[],
            early_lock_count=0,
            stage2_call_count=0,
            stage3_call_count=0,
        )

    selected = [
        candidate
        for candidate in record.candidates
        if candidate.detector_score >= policy.proposal_minimum_score
    ]
    decisions = [route_candidate(candidate, policy) for candidate in selected]
    object_decisions = [
        decision
        for decision in decisions
        if decision.outcome
        in {
            CandidateOutcome.APPROVED,
            CandidateOutcome.UNKNOWN,
            CandidateOutcome.SEGMENT_RECAPTURE,
        }
    ]
    reasons = []
    if any(decision.outcome is CandidateOutcome.UNRESOLVED for decision in decisions):
        reasons.append("STAGED_PROPOSAL_UNRESOLVED")
    if policy.require_exact_count:
        if record.scene.predicted_count is None:
            reasons.append("EXACT_COUNT_EVIDENCE_MISSING")
        elif record.scene.count_confidence < policy.count_confidence_minimum:
            reasons.append("EXACT_COUNT_LOW_CONFIDENCE")
        elif record.scene.predicted_count != len(object_decisions):
            reasons.append("EXACT_COUNT_MISMATCH")
    if policy.require_foreground_evidence:
        if record.scene.unexplained_foreground_score is None:
            reasons.append("FOREGROUND_EVIDENCE_MISSING")
        elif record.scene.unexplained_foreground_score > policy.unexplained_foreground_maximum:
            reasons.append("UNEXPLAINED_FOREGROUND")
    if not object_decisions:
        reasons.append("NO_RESOLVED_OBJECT")
    status: Literal["SEGMENTATION", "IMAGE_RECAPTURE"] = (
        "IMAGE_RECAPTURE" if reasons else "SEGMENTATION"
    )
    return ImageDecision(
        image_id=record.image_id,
        status=status,
        diagnostic_reasons=reasons,
        candidates=decisions,
        public_candidate_ids=(
            [] if status == "IMAGE_RECAPTURE" else [row.candidate_id for row in object_decisions]
        ),
        early_lock_count=sum(row.route is CandidateRoute.EARLY_LOCK for row in decisions),
        stage2_call_count=sum(row.used_stage2 for row in decisions),
        stage3_call_count=sum(row.used_stage3 for row in decisions),
    )


def evaluate_decisions(
    records: list[AdaptiveCascadeRecord],
    decisions: list[ImageDecision],
    policy: AdaptiveCascadePolicy,
) -> dict[str, Any]:
    if len(records) != len(decisions):
        raise ValueError("records and decisions are not aligned")
    totals: dict[str, int] = {
        "image_count": len(records),
        "ground_truth_object_count": 0,
        "stage1_proposal_miss_count": 0,
        "stage1_exact_proposal_image_count": 0,
        "segmentation_image_count": 0,
        "image_recapture_count": 0,
        "accepted_image_false_negative_count": 0,
        "accepted_object_false_negative_count": 0,
        "accepted_image_false_positive_count": 0,
        "accepted_object_false_positive_count": 0,
        "approved_count": 0,
        "approved_misrecognition_count": 0,
        "unknown_count": 0,
        "unknown_top3_candidate_out_count": 0,
        "segment_recapture_count": 0,
        "early_lock_count": 0,
        "early_lock_misrecognition_count": 0,
        "stage2_call_count": 0,
        "stage3_call_count": 0,
    }
    for record, image_decision in zip(records, decisions, strict=True):
        targets = {target.object_id: target.class_id for target in record.ground_truth}
        totals["ground_truth_object_count"] += len(targets)
        proposed = {
            candidate.matched_ground_truth_id
            for candidate in record.candidates
            if candidate.detector_score >= policy.proposal_minimum_score
            and candidate.matched_ground_truth_id is not None
        }
        proposal_misses = set(targets) - proposed
        totals["stage1_proposal_miss_count"] += len(proposal_misses)
        totals["stage1_exact_proposal_image_count"] += int(not proposal_misses)
        totals["early_lock_count"] += image_decision.early_lock_count
        totals["stage2_call_count"] += image_decision.stage2_call_count
        totals["stage3_call_count"] += image_decision.stage3_call_count
        for candidate in image_decision.candidates:
            if candidate.route is not CandidateRoute.EARLY_LOCK:
                continue
            target_class = targets.get(candidate.matched_ground_truth_id or "")
            totals["early_lock_misrecognition_count"] += int(
                target_class is None or candidate.predicted_class_id != target_class
            )
        if image_decision.status == "IMAGE_RECAPTURE":
            totals["image_recapture_count"] += 1
            continue
        totals["segmentation_image_count"] += 1
        public_ids = set(image_decision.public_candidate_ids)
        outputs = [
            candidate
            for candidate in image_decision.candidates
            if candidate.candidate_id in public_ids
        ]
        seen_targets: set[str] = set()
        false_positive_count = 0
        for candidate in outputs:
            target_id = candidate.matched_ground_truth_id
            if target_id is None or target_id in seen_targets:
                false_positive_count += 1
                continue
            seen_targets.add(target_id)
            expected_class = targets[target_id]
            if candidate.outcome is CandidateOutcome.APPROVED:
                totals["approved_count"] += 1
                totals["approved_misrecognition_count"] += int(
                    candidate.predicted_class_id != expected_class
                )
            elif candidate.outcome is CandidateOutcome.UNKNOWN:
                totals["unknown_count"] += 1
                totals["unknown_top3_candidate_out_count"] += int(
                    expected_class not in candidate.top3
                )
            elif candidate.outcome is CandidateOutcome.SEGMENT_RECAPTURE:
                totals["segment_recapture_count"] += 1
        false_negative_count = len(set(targets) - seen_targets)
        totals["accepted_object_false_negative_count"] += false_negative_count
        totals["accepted_image_false_negative_count"] += int(false_negative_count > 0)
        totals["accepted_object_false_positive_count"] += false_positive_count
        totals["accepted_image_false_positive_count"] += int(false_positive_count > 0)

    image_denominator = max(1, totals["image_count"])
    object_denominator = max(1, totals["ground_truth_object_count"])
    totals.update(
        {
            "segmentation_rate": totals["segmentation_image_count"] / image_denominator,
            "image_recapture_rate": totals["image_recapture_count"] / image_denominator,
            "stage1_exact_proposal_image_rate": (
                totals["stage1_exact_proposal_image_count"] / image_denominator
            ),
            "approved_rate": totals["approved_count"] / object_denominator,
            "unknown_rate": totals["unknown_count"] / object_denominator,
            "early_lock_rate": totals["early_lock_count"] / object_denominator,
        }
    )
    safety_counts = {
        "accepted_image_false_negative": totals["accepted_image_false_negative_count"],
        "accepted_image_false_positive": totals["accepted_image_false_positive_count"],
        "approved_misrecognition": totals["approved_misrecognition_count"],
        "unknown_top3_candidate_out": totals["unknown_top3_candidate_out_count"],
    }
    return {
        "metrics": totals,
        "zero_count_safety": {
            **safety_counts,
            "all_met": all(value == 0 for value in safety_counts.values()),
        },
    }


def load_records(path: Path) -> list[AdaptiveCascadeRecord]:
    records = [
        AdaptiveCascadeRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("adaptive cascade records are empty")
    image_ids = [str(record.image_id) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("adaptive cascade image_id values must be unique")
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = load_json_config(args.config)
    raw_policy = raw_config.get("adaptive_cascade")
    if not isinstance(raw_policy, dict):
        raise ValueError("configuration is missing adaptive_cascade")
    policy = AdaptiveCascadePolicy.model_validate(raw_policy)
    records = load_records(args.records)
    decisions = [decide_image(record, policy) for record in records]
    evaluation = evaluate_decisions(records, decisions, policy)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_adaptive_cascade",
        "split_role": args.split_role,
        "records": {
            "path": str(args.records.resolve()),
            "sha256": sha256_file(args.records),
        },
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "policy": policy.model_dump(mode="json"),
        **evaluation,
        "limitations": [
            "zero observed errors are not a per-image guarantee",
            "thresholds selected on development data require separate calibration and locked test",
            "model and ONNX provider parity are evaluated outside this policy simulator",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.decisions_output is not None:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        args.decisions_output.write_text(
            "".join(
                json.dumps(decision.model_dump(mode="json"), ensure_ascii=False) + "\n"
                for decision in decisions
            ),
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate recall-first adaptive cascade routing and zero-count safety"
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path)
    parser.add_argument(
        "--split-role",
        choices=("development", "calibration", "locked_test"),
        default="development",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
