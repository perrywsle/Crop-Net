"""Browser-first web application for TaoCrop."""

from crop_fusion_ai.web.app import create_app, main
from crop_fusion_ai.web.service import YieldModelService

__all__ = [
    "YieldModelService",
    "create_app",
    "main",
]
