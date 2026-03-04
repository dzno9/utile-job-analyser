import os
import unittest

from config import load_settings


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        for key in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(key, None)

    def test_default_provider_is_rule_based(self) -> None:
        settings = load_settings()
        self.assertEqual(settings.llm_provider, "rule_based")

    def test_invalid_provider_fails_fast(self) -> None:
        os.environ["LLM_PROVIDER"] = "invalid-provider"
        with self.assertRaises(ValueError):
            load_settings()

    def test_openai_provider_value_is_accepted(self) -> None:
        os.environ["LLM_PROVIDER"] = "openai"
        settings = load_settings()
        self.assertEqual(settings.llm_provider, "openai")


if __name__ == "__main__":
    unittest.main()
