from pathlib import Path

import handler


class NumpyLikeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class NumpyLikeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeOCR:
    def __call__(self, image_path, merge_level="paragraph"):
        assert image_path == "/tmp/page-1.png"
        assert merge_level == "paragraph"
        return [
            {
                "text": "안녕하세요",
                "confidence": 0.98,
                "left": 10,
                "upper": 20,
                "right": 110,
                "lower": 40,
                "quad": [[10, 20], [110, 20], [110, 40], [10, 40]],
            }
        ]


def test_handler_downloads_renders_and_returns_page_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(handler, "download_pdf", lambda url, timeout: b"%PDF-1.7")
    monkeypatch.setattr(
        handler,
        "render_pdf_pages",
        lambda pdf_bytes, page_ranges, dpi, output_dir: [
            handler.RenderedPage(
                page_number=1,
                image_path=Path("/tmp/page-1.png"),
                width=1700,
                height=2200,
            )
        ],
    )
    monkeypatch.setattr(handler, "get_ocr", lambda: FakeOCR())

    result = handler.handler(
        {
            "input": {
                "pdf_url": "https://example.com/sample.pdf",
                "pages": [[1, 1]],
                "dpi": 200,
                "merge_level": "paragraph",
            }
        }
    )

    assert result["model"] == "nvidia/nemotron-ocr-v2"
    assert result["variant"] == "v2_multilingual"
    assert result["pages"] == [
        {
            "page": 1,
            "width": 1700,
            "height": 2200,
            "predictions": [
                {
                    "text": "안녕하세요",
                    "confidence": 0.98,
                    "bbox": {
                        "left": 10,
                        "upper": 20,
                        "right": 110,
                        "lower": 40,
                    },
                    "quad": [[10, 20], [110, 20], [110, 40], [10, 40]],
                }
            ],
        }
    ]


def test_normalize_prediction_converts_numpy_like_values_to_json_safe_values():
    result = handler.normalize_prediction(
        {
            "text": "hello",
            "confidence": NumpyLikeScalar(0.9),
            "left": NumpyLikeScalar(1.5),
            "upper": NumpyLikeScalar(2.5),
            "right": NumpyLikeScalar(3.5),
            "lower": NumpyLikeScalar(4.5),
            "quad": NumpyLikeArray([[1.5, 2.5], [3.5, 4.5]]),
        }
    )

    assert result == {
        "text": "hello",
        "confidence": 0.9,
        "bbox": {
            "left": 1.5,
            "upper": 2.5,
            "right": 3.5,
            "lower": 4.5,
        },
        "quad": [[1.5, 2.5], [3.5, 4.5]],
    }
