from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationStatus = Literal[
    "already_exists",
    "possible_duplicate",
    "stored",
]


class EvaluationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_status: EvaluationStatus
    actual_status: EvaluationStatus

    expected_match_job_id: str | None = None
    matched_job_id: str | None = None

    retrieved_job_ids: list[str] = Field(default_factory=list)

    comparison_count: int = Field(ge=0)

    latency_seconds: float = Field(ge=0.0)

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
    )


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int
    correct_status_cases: int
    status_accuracy: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    duplicate_precision: float | None
    duplicate_recall: float | None

    retrieval_eligible_cases: int
    retrieval_hits_at_five: int
    retrieval_recall_at_five: float | None

    mean_candidate_rank: float | None
    mean_comparison_count: float

    false_positive_case_ids: list[str]
    false_negative_case_ids: list[str]

    total_latency_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float


DUPLICATE_STATUSES = {
    "already_exists",
    "possible_duplicate",
}


def summarize_evaluations(
    observations: list[EvaluationObservation],
) -> EvaluationSummary:
    if not observations:
        raise ValueError("At least one evaluation observation is required")

    correct_status_cases = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    retrieval_eligible_cases = 0
    retrieval_hits_at_five = 0
    candidate_ranks: list[int] = []

    false_positive_case_ids: list[str] = []
    false_negative_case_ids: list[str] = []

    for observation in observations:
        if observation.actual_status == observation.expected_status:
            correct_status_cases += 1

        expected_duplicate = observation.expected_status in DUPLICATE_STATUSES
        actual_duplicate = observation.actual_status in DUPLICATE_STATUSES

        if expected_duplicate and actual_duplicate:
            true_positives += 1
        elif not expected_duplicate and actual_duplicate:
            false_positives += 1
            false_positive_case_ids.append(observation.case_id)
        elif expected_duplicate and not actual_duplicate:
            false_negatives += 1
            false_negative_case_ids.append(observation.case_id)
        else:
            true_negatives += 1

        if (
            observation.expected_status == "possible_duplicate"
            and observation.expected_match_job_id is not None
        ):
            retrieval_eligible_cases += 1

            if observation.expected_match_job_id in observation.retrieved_job_ids:
                retrieval_hits_at_five += 1

                candidate_ranks.append(
                    observation.retrieved_job_ids.index(
                        observation.expected_match_job_id
                    )
                    + 1
                )

    predicted_duplicate_cases = true_positives + false_positives
    expected_duplicate_cases = true_positives + false_negatives

    duplicate_precision = (
        true_positives / predicted_duplicate_cases
        if predicted_duplicate_cases
        else None
    )

    duplicate_recall = (
        true_positives / expected_duplicate_cases if expected_duplicate_cases else None
    )

    retrieval_recall_at_five = (
        retrieval_hits_at_five / retrieval_eligible_cases
        if retrieval_eligible_cases
        else None
    )

    mean_candidate_rank = (
        sum(candidate_ranks) / len(candidate_ranks) if candidate_ranks else None
    )

    total_cases = len(observations)

    return EvaluationSummary(
        total_cases=total_cases,
        correct_status_cases=correct_status_cases,
        status_accuracy=(correct_status_cases / total_cases),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        duplicate_precision=duplicate_precision,
        duplicate_recall=duplicate_recall,
        retrieval_eligible_cases=(retrieval_eligible_cases),
        retrieval_hits_at_five=(retrieval_hits_at_five),
        retrieval_recall_at_five=(retrieval_recall_at_five),
        mean_candidate_rank=mean_candidate_rank,
        mean_comparison_count=(
            sum(observation.comparison_count for observation in observations)
            / total_cases
        ),
        false_positive_case_ids=(false_positive_case_ids),
        false_negative_case_ids=(false_negative_case_ids),
        total_latency_seconds=sum(
            observation.latency_seconds for observation in observations
        ),
        total_input_tokens=sum(
            observation.input_tokens or 0 for observation in observations
        ),
        total_output_tokens=sum(
            observation.output_tokens or 0 for observation in observations
        ),
        total_estimated_cost_usd=sum(
            observation.estimated_cost_usd or 0.0 for observation in observations
        ),
    )


EVALUATION_NAMESPACE_PREFIX = "evaluation-"


def validate_evaluation_namespace(
    namespace: str,
) -> str:
    normalized_namespace = namespace.strip()

    if not normalized_namespace.startswith(EVALUATION_NAMESPACE_PREFIX):
        raise ValueError(
            f"Evaluation namespace must start with {EVALUATION_NAMESPACE_PREFIX!r}"
        )

    if normalized_namespace == EVALUATION_NAMESPACE_PREFIX:
        raise ValueError("Evaluation namespace must include a unique suffix")

    return normalized_namespace
