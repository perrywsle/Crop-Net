"""Thin optional wrapper around the official CropNet package APIs."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, Field

from crop_fusion_ai.config.schemas import ImagePrediction, YieldInput
from crop_fusion_ai.data_sources.cropnet_schemas import CropNetQuery, CropNetSample


class CropNetDependencyError(ImportError):
    """Raised when the optional official CropNet package is unavailable."""


class CropNetDownloadResult(BaseModel):
    """Summary of a bounded CropNet API download request."""

    target_dir: Path
    requested_modalities: list[str] = Field(default_factory=list)
    fips_codes: list[str]
    years: list[int]


class _DataDownloader(Protocol):
    """Protocol for the official CropNet DataDownloader API used here."""

    def download_USDA(
        self,
        crop_type: str,
        *,
        fips_codes: list[str],
        years: list[str],
    ) -> object:
        """Download USDA crop records."""

    def download_HRRR(self, *, fips_codes: list[str], years: list[str]) -> object:
        """Download WRF-HRRR weather records."""

    def download_Sentinel2(
        self,
        *,
        fips_codes: list[str],
        years: list[str],
        image_type: str,
    ) -> object:
        """Download Sentinel-2 imagery records."""


class _DataDownloaderFactory(Protocol):
    """Factory protocol for constructing a CropNet downloader."""

    def __call__(self, *, target_dir: str) -> _DataDownloader:
        """Create a downloader for the given target directory."""


class _DataDownloaderModule(Protocol):
    """Protocol for the CropNet module containing DataDownloader."""

    DataDownloader: _DataDownloaderFactory


ImportModule = Callable[[str], object]


class CropNetClient:
    """Client for small, selective CropNet data requests.

    The official ``cropnet`` dependency is imported lazily so normal unit tests,
    type checks, and UI demos can run without installing the heavy data package.
    """

    def __init__(
        self,
        target_dir: str | Path = Path("data/cropnet_cache"),
        *,
        downloader_factory: _DataDownloaderFactory | None = None,
        import_module_func: ImportModule = import_module,
    ) -> None:
        """Create a CropNet client that writes API results under ``target_dir``."""
        self.target_dir = Path(target_dir)
        self._downloader_factory = downloader_factory
        self._import_module_func = import_module_func

    def download_query(self, query: CropNetQuery) -> CropNetDownloadResult:
        """Run a bounded CropNet download for the selected query modalities."""
        downloader = self._create_downloader()
        years = query.years_as_strings()
        requested_modalities: list[str] = []

        if query.include_usda:
            downloader.download_USDA(
                query.crop_type,
                fips_codes=query.fips_codes,
                years=years,
            )
            requested_modalities.append("USDA")

        if query.include_hrrr:
            downloader.download_HRRR(fips_codes=query.fips_codes, years=years)
            requested_modalities.append("HRRR")

        if query.include_sentinel2:
            downloader.download_Sentinel2(
                fips_codes=query.fips_codes,
                years=years,
                image_type=query.image_type,
            )
            requested_modalities.append(f"Sentinel2:{query.image_type}")

        return CropNetDownloadResult(
            target_dir=self.target_dir,
            requested_modalities=requested_modalities,
            fips_codes=query.fips_codes,
            years=query.years,
        )

    def _create_downloader(self) -> _DataDownloader:
        """Create the official CropNet downloader or raise a setup error."""
        if self._downloader_factory is not None:
            return self._downloader_factory(target_dir=str(self.target_dir))

        try:
            module = self._import_module_func("cropnet.data_downloader")
        except ImportError as exc:
            msg = (
                "The optional 'cropnet' package is required for real CropNet API "
                "downloads. Install it in a compatible environment before using "
                "CropNetClient.download_query()."
            )
            raise CropNetDependencyError(msg) from exc

        data_downloader_module = cast(_DataDownloaderModule, module)
        return data_downloader_module.DataDownloader(target_dir=str(self.target_dir))


def cropnet_sample_to_yield_input(
    sample: CropNetSample,
    image_prediction: ImagePrediction,
) -> YieldInput:
    """Combine a normalized CropNet sample with image-model output."""
    from crop_fusion_ai.config.schemas import CropFeatures

    return YieldInput(
        weather=sample.weather,
        crop=CropFeatures(
            crop_type=sample.crop_type,
            region=sample.region or sample.fips_code,
            year=sample.year,
            planting_age_days=None,
        ),
        image_prediction=image_prediction,
    )
