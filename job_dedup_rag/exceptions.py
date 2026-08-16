class ExternalServiceOperationError(RuntimeError):
    def __init__(
        self,
        operation: str,
        job_id: str,
    ) -> None:
        self.operation = operation
        self.job_id = job_id

        super().__init__(
            f"External service operation {operation!r} failed for job ID {job_id!r}"
        )
