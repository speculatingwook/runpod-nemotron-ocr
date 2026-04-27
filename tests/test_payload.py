import pytest

from handler import InputError, parse_payload


def test_parse_payload_accepts_pdf_url_and_page_ranges():
    request = parse_payload(
        {
            "pdf_url": "https://example.com/sample.pdf",
            "pages": [[1, 6], [10, 10]],
            "dpi": 200,
            "merge_level": "paragraph",
        }
    )

    assert request.pdf_url == "https://example.com/sample.pdf"
    assert request.pdf_base64 is None
    assert request.page_ranges == [(1, 6), (10, 10)]
    assert request.dpi == 200
    assert request.merge_level == "paragraph"


def test_parse_payload_requires_one_pdf_source():
    with pytest.raises(InputError, match="Provide exactly one"):
        parse_payload({})


def test_parse_payload_rejects_descending_page_range():
    with pytest.raises(InputError, match="start <= end"):
        parse_payload(
            {
                "pdf_url": "https://example.com/sample.pdf",
                "pages": [[6, 1]],
            }
        )
