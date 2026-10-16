from dataclasses import dataclass


@dataclass(frozen=True)
class ContentSource:
    """Describe one file or container to resolve."""

    name: str
    path: str | None = None
    type: str | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        normalized_path = self.path.strip() if self.path else None
        normalized_type = (
            self.type.strip().lower().lstrip(".")
            if self.type
            else None
        )

        if not normalized_name:
            raise ValueError("ContentSource.name must not be empty.")

        if normalized_type == "":
            normalized_type = None

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "type", normalized_type)