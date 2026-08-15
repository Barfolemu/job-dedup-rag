from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractedJobFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str | None = Field(
        description="Company offering the position, or null if unavailable"
    )
    role_title: str | None = Field(description="Job title, or null if unavailable")
    requisition_id: str | None = Field(
        description="Employer requisition or posting ID, or null if unavailable"
    )
    location: str | None = Field(
        description="Location stated in the job description, or null"
    )
    workplace_type: Literal[
        "remote",
        "hybrid",
        "onsite",
        "unspecified",
    ] = Field(description="Where the employee is expected to work")
    employment_type: Literal[
        "full_time",
        "part_time",
        "contract",
        "temporary",
        "internship",
        "unspecified",
    ] = Field(
        description=(
            "Employment arrangement explicitly stated in the job description. "
            "Return unspecified unless the description directly identifies the "
            "position as full-time, part-time, contract, temporary, or internship. "
            "Do not infer full-time from salary, benefits, or the nature of the role."
        )
    )
    seniority_level: str | None = Field(
        description="Seniority or management level, or null if unavailable"
    )
    team_or_domain: str | None = Field(
        description="Team, product, or business domain, or null if unavailable"
    )
    responsibilities: list[str] = Field(
        description="Concise list of the role's primary responsibilities"
    )
    required_qualifications: list[str] = Field(
        description="Concise list of required qualifications"
    )
    preferred_qualifications: list[str] = Field(
        description="Concise list of preferred qualifications"
    )
    technologies: list[str] = Field(
        description="Technologies, platforms, and technical practices mentioned"
    )

    def to_search_text(self) -> str:
        def format_list(label: str, values: list[str]) -> str:
            items = "\n".join(f"- {value}" for value in values)
            return f"{label}:\n{items or '- None stated'}"

        sections = [
            f"Company: {self.company_name or 'Unspecified'}",
            f"Role title: {self.role_title or 'Unspecified'}",
            f"Requisition ID: {self.requisition_id or 'Unspecified'}",
            f"Location: {self.location or 'Unspecified'}",
            f"Workplace type: {self.workplace_type}",
            f"Employment type: {self.employment_type}",
            f"Seniority level: {self.seniority_level or 'Unspecified'}",
            f"Team or domain: {self.team_or_domain or 'Unspecified'}",
            format_list("Responsibilities", self.responsibilities),
            format_list(
                "Required qualifications",
                self.required_qualifications,
            ),
            format_list(
                "Preferred qualifications",
                self.preferred_qualifications,
            ),
            format_list("Technologies", self.technologies),
        ]

        return "\n\n".join(sections)


class DuplicateComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_duplicate: bool = Field(
        description=(
            "True only when the two job descriptions represent the same "
            "underlying position, not merely similar roles"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=("Confidence in the duplicate decision, from 0.0 to 1.0"),
    )
    matching_signals: list[str] = Field(
        description=("Specific facts supporting that the positions are the same")
    )
    differences: list[str] = Field(
        description=("Meaningful differences between the two positions")
    )
    explanation: str = Field(
        description=(
            "Concise explanation of why the positions are or are not duplicates"
        )
    )
