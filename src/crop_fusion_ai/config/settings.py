"""Application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Validated filesystem paths used by the project."""

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CROP_FUSION_",
        extra="ignore",
    )

    def resolve_path(self, path: Path) -> Path:
        """Return an absolute path relative to the project root when needed."""
        if path.is_absolute():
            return path
        return self.project_root / path
