from __future__ import annotations

from pathlib import Path


class CVParser:
    """Extract plain text from a CV PDF using fitz, with pdfplumber fallback."""

    def parse(self, pdf_path: str | Path) -> str:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"CV PDF not found: {path}")

        text = self._parse_with_fitz(path)
        if text.strip():
            return text

        text = self._parse_with_pdfplumber(path)
        if text.strip():
            return text

        raise ValueError(
            "Could not extract text from CV PDF with fitz or pdfplumber. "
            "The PDF is likely scanned/image-based or copy-protected."
        )

    def _parse_with_fitz(self, path: Path) -> str:
        try:
            import fitz  # type: ignore
        except ImportError:
            return ""

        chunks: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                chunks.append(page.get_text("text") or "")
        return "\n".join(chunks).strip()

    def _parse_with_pdfplumber(self, path: Path) -> str:
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return ""

        chunks: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
        return "\n".join(chunks).strip()
