from job_dedup_rag.evaluation import (
    EvaluationObservation,
    summarize_evaluations,
    validate_evaluation_namespace,
)

observations = [
    EvaluationObservation(
        case_id="exact-id",
        expected_status="already_exists",
        actual_status="already_exists",
        expected_match_job_id="linkedin:exact",
        matched_job_id="linkedin:exact",
        comparison_count=0,
        latency_seconds=0.2,
    ),
    EvaluationObservation(
        case_id="semantic-duplicate",
        expected_status="possible_duplicate",
        actual_status="possible_duplicate",
        expected_match_job_id="candidate:match",
        matched_job_id="candidate:match",
        retrieved_job_ids=[
            "candidate:similar",
            "candidate:match",
        ],
        comparison_count=2,
        latency_seconds=4.0,
        input_tokens=1000,
        output_tokens=100,
        estimated_cost_usd=0.003,
    ),
    EvaluationObservation(
        case_id="unique-false-positive",
        expected_status="stored",
        actual_status="possible_duplicate",
        matched_job_id="candidate:generic",
        retrieved_job_ids=[
            "candidate:generic",
        ],
        comparison_count=1,
        latency_seconds=3.0,
        input_tokens=800,
        output_tokens=80,
        estimated_cost_usd=0.002,
    ),
    EvaluationObservation(
        case_id="semantic-false-negative",
        expected_status="possible_duplicate",
        actual_status="stored",
        expected_match_job_id="candidate:missing",
        retrieved_job_ids=[
            "candidate:one",
            "candidate:two",
        ],
        comparison_count=5,
        latency_seconds=5.0,
        input_tokens=2000,
        output_tokens=200,
        estimated_cost_usd=0.006,
    ),
    EvaluationObservation(
        case_id="unique-correct",
        expected_status="stored",
        actual_status="stored",
        retrieved_job_ids=[
            "candidate:one",
            "candidate:two",
        ],
        comparison_count=5,
        latency_seconds=2.0,
        input_tokens=700,
        output_tokens=70,
        estimated_cost_usd=0.0015,
    ),
]

summary = summarize_evaluations(observations)

assert summary.total_cases == 5
assert summary.correct_status_cases == 3
assert summary.status_accuracy == 0.6

assert summary.true_positives == 2
assert summary.false_positives == 1
assert summary.true_negatives == 1
assert summary.false_negatives == 1

assert summary.duplicate_precision == 2 / 3
assert summary.duplicate_recall == 2 / 3

assert summary.retrieval_eligible_cases == 2
assert summary.retrieval_hits_at_five == 1
assert summary.retrieval_recall_at_five == 0.5
assert summary.mean_candidate_rank == 2.0

assert summary.mean_comparison_count == 2.6

assert summary.false_positive_case_ids == ["unique-false-positive"]
assert summary.false_negative_case_ids == ["semantic-false-negative"]

assert summary.total_latency_seconds == 14.2
assert summary.total_input_tokens == 4500
assert summary.total_output_tokens == 450
assert (
    round(
        summary.total_estimated_cost_usd,
        4,
    )
    == 0.0125
)

print("Status accuracy:", summary.status_accuracy)
print(
    "Duplicate precision:",
    summary.duplicate_precision,
)
print(
    "Duplicate recall:",
    summary.duplicate_recall,
)
print(
    "Retrieval recall at five:",
    summary.retrieval_recall_at_five,
)
print(
    "Mean candidate rank:",
    summary.mean_candidate_rank,
)
print(
    "Mean comparisons:",
    summary.mean_comparison_count,
)
print("Evaluation summary checks passed")

assert validate_evaluation_namespace("  evaluation-test-run  ") == "evaluation-test-run"

unsafe_namespaces = [
    "",
    "structured-v1",
    "production",
    "evaluation-",
]

for unsafe_namespace in unsafe_namespaces:
    try:
        validate_evaluation_namespace(unsafe_namespace)
    except ValueError as error:
        print(
            "Expected namespace error:",
            error,
        )
    else:
        raise AssertionError(
            f"Unsafe evaluation namespace was accepted: {unsafe_namespace!r}"
        )

print("Evaluation namespace checks passed")
