"""Crop-and-zoom preprocessing for chart/figure extraction.

Chart extraction is a known weak spot for vision models: a chart's axis
labels and gridlines are small relative to a full page, so the model must
both LOCATE and READ the chart in one pass. Cropping to the figure's
bounding box and upscaling turns that into just READ — the highest-leverage
fix before reaching for a different model or a chart-digitisation tool.
(Adopted from the uploaded figure_preprocessor; the design is unchanged.)

What differs here from the upload: the upload left its page-image source
(`DatabricksPageImageProvider`) as ``NotImplementedError``, so crop-and-zoom
never actually ran in their production. This package already rasterises PDF
pages with pdfplumber (in the OCR fallback) and depends on Pillow, so we
supply a WORKING provider — ``PdfPlumberPageImageProvider`` — and the
technique runs for real on PDF sources. An Azure/Databricks provider can be
added later for non-PDF page renders.

IMPORTANT: the image handed to the MODEL is clean — no highlight box drawn
on it (that would bias the model's own reading). Drawing a visible box for a
HUMAN reviewer is a separate concern (see reviewer/bbox rendering).
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

DEFAULT_PADDING = 0.02            # fraction of page dim added around bbox each side
DEFAULT_MIN_OUTPUT_DIMENSION = 1024  # upscale smaller cropped dim to at least this
DEFAULT_MAX_OUTPUT_DIMENSION = 2000  # cap to bound image size / token cost


class PageImageProvider(ABC):
    """Abstraction over where a rendered page image comes from."""

    @abstractmethod
    def get_page_image(self, document_bytes: bytes, filename: str,
                       page_number: int):
        """Return a PIL.Image for the page, or None if unavailable."""


class PdfPlumberPageImageProvider(PageImageProvider):
    """Working provider for PDF sources, backed by pdfplumber's page renderer.

    This is the piece the uploaded codebase left unimplemented. pdfplumber is
    already a dependency (used by the preprocessor / OCR fallback), so no new
    requirement is introduced. Returns None for non-PDF inputs or if the page
    can't be rendered, so callers degrade to text-only rather than fail.
    """

    def __init__(self, resolution: int = 200):
        self._resolution = resolution

    def get_page_image(self, document_bytes: bytes, filename: str,
                       page_number: int):
        if not filename.lower().endswith(".pdf"):
            return None
        try:
            import pdfplumber  # lazy
            from PIL import Image  # noqa: F401 (ensures Pillow present)

            with pdfplumber.open(BytesIO(document_bytes)) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                page = pdf.pages[page_number - 1]
                pil = page.to_image(resolution=self._resolution).original
                return pil
        except Exception:
            return None


def crop_and_zoom(page_image, bbox, padding: float = DEFAULT_PADDING,
                  min_output_dimension: int = DEFAULT_MIN_OUTPUT_DIMENSION,
                  max_output_dimension: int = DEFAULT_MAX_OUTPUT_DIMENSION):
    """Crop ``page_image`` to ``bbox`` (normalised [x0,y0,x1,y1], 0-1) plus
    padding, then upscale so the smaller cropped dimension reaches
    ``min_output_dimension`` px — never past ``max_output_dimension``.

    Padding matters: charts often have axis labels or legends just outside
    the plot area, so cropping tight to the bbox risks cutting off the labels
    needed to read the chart.
    """
    from PIL import Image

    width, height = page_image.size
    x0, y0, x1, y1 = bbox

    px0 = int(max(0.0, x0 - padding) * width)
    py0 = int(max(0.0, y0 - padding) * height)
    px1 = int(min(1.0, x1 + padding) * width)
    py1 = int(min(1.0, y1 + padding) * height)
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)  # guard zero-area crop

    cropped = page_image.crop((px0, py0, px1, py1))
    crop_w, crop_h = cropped.size
    smaller = min(crop_w, crop_h)

    scale = 1.0
    if smaller < min_output_dimension:
        scale = min_output_dimension / smaller
    if max(crop_w, crop_h) * scale > max_output_dimension:
        scale = max_output_dimension / max(crop_w, crop_h)

    if scale != 1.0:
        new_size = (max(1, round(crop_w * scale)), max(1, round(crop_h * scale)))
        cropped = cropped.resize(new_size, Image.LANCZOS)
    return cropped


def encode_image_base64(image, fmt: str = "PNG") -> str:
    """Base64-encode a PIL image for a multimodal LLM image block."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prepare_figure_image(provider: PageImageProvider, document_bytes: bytes,
                         filename: str, page_number: Optional[int],
                         bbox) -> Optional[object]:
    """Full crop-and-zoom for one figure element. Returns a PIL image, or None
    if there's no bbox/page to crop (caller falls back to text-only, never
    fails)."""
    if bbox is None or page_number is None:
        return None
    page_image = provider.get_page_image(document_bytes, filename, page_number)
    if page_image is None:
        return None
    return crop_and_zoom(page_image, bbox)


_CHART_KEYWORDS = None


def is_chart_like(element) -> bool:
    """Heuristic: a FIGURE element whose parser description mentions a chart.

    Kept here (not just in routing) so the extractor can gate the crop-zoom
    path on exactly the same signal the router uses.
    """
    import re
    from ..shared.schema import Modality

    global _CHART_KEYWORDS
    if _CHART_KEYWORDS is None:
        _CHART_KEYWORDS = re.compile(
            r"\b(chart|graph|plot|trend line|bar chart|pie chart|line graph|"
            r"histogram|axis)\b", re.IGNORECASE)

    modality = element.modality
    modality = modality.value if isinstance(modality, Modality) else str(modality)
    if modality != Modality.FIGURE.value:
        return False
    return bool(element.description and _CHART_KEYWORDS.search(element.description))
