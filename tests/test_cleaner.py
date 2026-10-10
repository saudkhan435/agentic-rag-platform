from app.ingestion.cleaner import clean_text


def test_joins_hyphenated_words():
    text = "secu-\nrity vulnerabilities"

    result = clean_text(text)

    assert result == "security vulnerabilities"


def test_replaces_single_line_breaks():
    text = "Agentic systems\nuse external tools."

    result = clean_text(text)

    assert result == "Agentic systems use external tools."


def test_removes_repeated_spaces():
    text = "Agentic     AI     systems"

    result = clean_text(text)

    assert result == "Agentic AI systems"