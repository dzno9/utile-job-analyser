from __future__ import annotations

import unittest
from unittest import mock

import app
from stages.ui_orchestrator import PipelineResult


class RerunSignal(RuntimeError):
    pass


class FakeStreamlit:
    def __init__(self, session_state: dict) -> None:
        self.session_state = session_state

    def container(self):
        return object()

    def rerun(self) -> None:
        raise RerunSignal()


class FakeOrchestrator:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result

    def run(self, request, *, on_event=None):  # noqa: ANN001
        del request, on_event
        return self.result


class TestAppFlow(unittest.TestCase):
    def _base_state(self) -> dict:
        return {
            "job_url": "https://jobs.example.com/acme/data-analyst",
            "candidate_notes": "notes",
            "cv_bytes": b"pdf",
            "analysis_result": None,
            "events": [],
            "current_view": app.VIEW_INPUT,
            "scrape_failed": False,
            "cv_parse_failed": False,
            "manual_job_text": "",
            "manual_cv_text": "",
        }

    def test_execute_pipeline_reroutes_to_inline_job_text_on_scrape_failure(self) -> None:
        fake_st = FakeStreamlit(self._base_state())
        result = PipelineResult(needs_manual_posting_text=True, needs_manual_cv_text=False)
        with (
            mock.patch.object(app, "st", fake_st),
            mock.patch.object(app, "build_orchestrator", return_value=FakeOrchestrator(result)),
        ):
            with self.assertRaises(RerunSignal):
                app._execute_pipeline()

        self.assertTrue(fake_st.session_state["scrape_failed"])
        self.assertFalse(fake_st.session_state["cv_parse_failed"])
        self.assertEqual(fake_st.session_state["current_view"], app.VIEW_INPUT)

    def test_execute_pipeline_reroutes_to_inline_cv_text_on_parse_failure(self) -> None:
        fake_st = FakeStreamlit(self._base_state())
        result = PipelineResult(needs_manual_posting_text=False, needs_manual_cv_text=True)
        with (
            mock.patch.object(app, "st", fake_st),
            mock.patch.object(app, "build_orchestrator", return_value=FakeOrchestrator(result)),
        ):
            with self.assertRaises(RerunSignal):
                app._execute_pipeline()

        self.assertFalse(fake_st.session_state["scrape_failed"])
        self.assertTrue(fake_st.session_state["cv_parse_failed"])
        self.assertEqual(fake_st.session_state["current_view"], app.VIEW_INPUT)

    def test_execute_pipeline_success_clears_inline_error_flags(self) -> None:
        fake_st = FakeStreamlit(self._base_state())
        fake_st.session_state["scrape_failed"] = True
        fake_st.session_state["cv_parse_failed"] = True
        fake_st.session_state["manual_job_text"] = "Manual job"
        fake_st.session_state["manual_cv_text"] = "Manual CV"
        result = PipelineResult(needs_manual_posting_text=False, needs_manual_cv_text=False)
        with (
            mock.patch.object(app, "st", fake_st),
            mock.patch.object(app, "build_orchestrator", return_value=FakeOrchestrator(result)),
        ):
            with self.assertRaises(RerunSignal):
                app._execute_pipeline()

        self.assertFalse(fake_st.session_state["scrape_failed"])
        self.assertFalse(fake_st.session_state["cv_parse_failed"])
        self.assertEqual(fake_st.session_state["manual_job_text"], "Manual job")
        self.assertEqual(fake_st.session_state["manual_cv_text"], "Manual CV")
        self.assertEqual(fake_st.session_state["current_view"], app.VIEW_RESULTS)


if __name__ == "__main__":
    unittest.main()
