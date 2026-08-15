from job_dedup_rag.vector_store import job_id_exists

existing_job_id = "linkedin:4413856088"
missing_job_id = "test:exact-id-that-should-not-exist"

existing_found = job_id_exists(existing_job_id)
missing_found = job_id_exists(missing_job_id)

assert existing_found
assert not missing_found

print("Existing ID found:", existing_found)
print("Missing ID found:", missing_found)
