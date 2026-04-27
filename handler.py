import base64
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/nemotron-ocr-v2"
VARIANT = "v2_multilingual"
DEFAULT_DPI = 200
DEFAULT_MERGE_LEVEL = "paragraph"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DPI = 400
MIN_DPI = 72
MAX_PAGES_PER_REQUEST = 100
VALID_MERGE_LEVELS = {"word", "sentence", "paragraph"}


class InputError(ValueError):
    """Raised when a request payload cannot be processed."""


@dataclass(frozen=True)
class OcrRequest:
    pdf_url: str | None
    pdf_base64: str | None
    page_ranges: list[tuple[int, int]] | None
    dpi: int
    merge_level: str
    download_timeout: int


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path
    width: int
    height: int


ocr = None


def parse_payload(payload: dict[str, Any]) -> OcrRequest:
    if not isinstance(payload, dict):
        raise InputError("Request input must be a JSON object.")

    pdf_url = payload.get("pdf_url")
    pdf_base64 = payload.get("pdf_base64")
    if bool(pdf_url) == bool(pdf_base64):
        raise InputError("Provide exactly one of pdf_url or pdf_base64.")

    dpi = _parse_int(payload.get("dpi", DEFAULT_DPI), "dpi")
    if dpi < MIN_DPI or dpi > MAX_DPI:
        raise InputError(f"dpi must be between {MIN_DPI} and {MAX_DPI}.")

    merge_level = payload.get("merge_level", DEFAULT_MERGE_LEVEL)
    if merge_level not in VALID_MERGE_LEVELS:
        levels = ", ".join(sorted(VALID_MERGE_LEVELS))
        raise InputError(f"merge_level must be one of: {levels}.")

    timeout = _parse_int(
        payload.get("download_timeout", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS),
        "download_timeout",
    )
    if timeout <= 0:
        raise InputError("download_timeout must be greater than 0.")

    page_ranges = _parse_page_ranges(payload.get("pages"))
    return OcrRequest(
        pdf_url=pdf_url,
        pdf_base64=pdf_base64,
        page_ranges=page_ranges,
        dpi=dpi,
        merge_level=merge_level,
        download_timeout=timeout,
    )


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise InputError(f"{name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be an integer.") from exc


def _parse_page_ranges(value: Any) -> list[tuple[int, int]] | None:
    if value in (None, []):
        return None
    if not isinstance(value, list):
        raise InputError("pages must be a list of [start, end] ranges.")

    ranges: list[tuple[int, int]] = []
    requested_page_count = 0
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise InputError("Each pages entry must be [start, end].")
        start = _parse_int(item[0], "page start")
        end = _parse_int(item[1], "page end")
        if start < 1 or end < 1:
            raise InputError("Page numbers are 1-based and must be positive.")
        if start > end:
            raise InputError("Each page range must satisfy start <= end.")
        requested_page_count += end - start + 1
        ranges.append((start, end))

    if requested_page_count > MAX_PAGES_PER_REQUEST:
        raise InputError(
            f"Requested page count must be <= {MAX_PAGES_PER_REQUEST} pages."
        )
    return ranges


def get_ocr():
    global ocr
    if ocr is not None:
        return ocr

    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

    model_dir = os.getenv("NEMOTRON_MODEL_DIR")
    if model_dir:
        ocr = NemotronOCRV2(model_dir=model_dir)
    else:
        ocr = NemotronOCRV2(lang=os.getenv("NEMOTRON_LANG", "multi"))
    return ocr


def load_pdf_bytes(request: OcrRequest) -> bytes:
    if request.pdf_url:
        return download_pdf(request.pdf_url, timeout=request.download_timeout)
    if request.pdf_base64:
        return decode_pdf_base64(request.pdf_base64)
    raise InputError("Missing PDF source.")


def download_pdf(url: str, timeout: int) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type and "pdf" not in content_type.lower():
        # Some object stores return application/octet-stream, so keep this as a hint.
        if "octet-stream" not in content_type.lower():
            raise InputError(f"URL did not return a PDF-like content type: {content_type}")
    return response.content


def decode_pdf_base64(value: str) -> bytes:
    if "," in value and value.strip().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise InputError("pdf_base64 is not valid base64.") from exc


def render_pdf_pages(
    pdf_bytes: bytes,
    page_ranges: list[tuple[int, int]] | None,
    dpi: int,
    output_dir: Path,
) -> list[RenderedPage]:
    import fitz

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        ranges = page_ranges or [(1, document.page_count)]
        page_numbers = _expand_page_ranges(ranges, document.page_count)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        rendered_pages: list[RenderedPage] = []

        for page_number in page_numbers:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page-{page_number}.png"
            pixmap.save(str(image_path))
            rendered_pages.append(
                RenderedPage(
                    page_number=page_number,
                    image_path=image_path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
        return rendered_pages
    finally:
        document.close()


def _expand_page_ranges(
    ranges: list[tuple[int, int]],
    page_count: int,
) -> list[int]:
    page_numbers: list[int] = []
    for start, end in ranges:
        if end > page_count:
            raise InputError(f"Requested page {end}, but PDF has {page_count} pages.")
        page_numbers.extend(range(start, end + 1))
    if len(page_numbers) > MAX_PAGES_PER_REQUEST:
        raise InputError(f"Requested page count must be <= {MAX_PAGES_PER_REQUEST} pages.")
    return page_numbers


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": _json_safe_value(prediction.get("text", "")),
        "confidence": _json_safe_value(prediction.get("confidence")),
        "bbox": {
            "left": _json_safe_value(prediction.get("left")),
            "upper": _json_safe_value(prediction.get("upper")),
            "right": _json_safe_value(prediction.get("right")),
            "lower": _json_safe_value(prediction.get("lower")),
        },
        "quad": _json_safe_value(prediction.get("quad")),
    }


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_safe_value(value.tolist())
    if hasattr(value, "item"):
        return _json_safe_value(value.item())
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return value


def run_ocr_on_pages(
    rendered_pages: list[RenderedPage],
    merge_level: str,
) -> list[dict[str, Any]]:
    model = get_ocr()
    pages: list[dict[str, Any]] = []
    for rendered_page in rendered_pages:
        predictions = model(str(rendered_page.image_path), merge_level=merge_level)
        pages.append(
            {
                "page": rendered_page.page_number,
                "width": rendered_page.width,
                "height": rendered_page.height,
                "predictions": [
                    normalize_prediction(prediction) for prediction in predictions
                ],
            }
        )
    return pages


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = event["input"]
        request = parse_payload(payload)
        pdf_bytes = load_pdf_bytes(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            rendered_pages = render_pdf_pages(
                pdf_bytes=pdf_bytes,
                page_ranges=request.page_ranges,
                dpi=request.dpi,
                output_dir=Path(temp_dir),
            )
            pages = run_ocr_on_pages(rendered_pages, request.merge_level)

        return {
            "model": MODEL_ID,
            "variant": VARIANT,
            "dpi": request.dpi,
            "merge_level": request.merge_level,
            "pages": pages,
        }
    except KeyError as exc:
        raise InputError("RunPod event must contain input.") from exc


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
