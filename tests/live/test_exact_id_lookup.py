from job_dedup_rag.vector_store import job_id_exists


def test_job_id_exists_distinguishes_existing_from_missing() -> None:
    """Requires the production namespace to already contain linkedin:4413856088."""
    assert job_id_exists("linkedin:4413856088")
    assert not job_id_exists("test:exact-id-that-should-not-exist")
