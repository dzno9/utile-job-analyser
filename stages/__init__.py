from .job_posting_scraper import (
    FetchedPage,
    HttpxURLFetcher,
    JobPostingScraper,
    ScrapeResult,
    detect_platform,
)
from .gap_analyzer import GapAnalyzer, GapAnalyzerResult
from .application_materials_generator import (
    ApplicationMaterialsGenerator,
    ApplicationMaterialsResult,
    build_materials_generator,
)
from .ui_orchestrator import PipelineOrchestrator, PipelineRequest, PipelineResult, StageEvent

__all__ = [
    "GapAnalyzer",
    "GapAnalyzerResult",
    "ApplicationMaterialsGenerator",
    "ApplicationMaterialsResult",
    "build_materials_generator",
    "FetchedPage",
    "HttpxURLFetcher",
    "JobPostingScraper",
    "ScrapeResult",
    "detect_platform",
    "PipelineOrchestrator",
    "PipelineRequest",
    "PipelineResult",
    "StageEvent",
]
