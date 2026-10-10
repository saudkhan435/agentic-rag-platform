import re


def clean_text(text: str) -> str:
    # Join words that were split across PDF line breaks:
    # "secu-\nrity" -> "security"
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text
    )

    # Replace single line breaks with spaces.
    # PDF extraction often creates one newline per printed line.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text
    )

    # Keep paragraph separation but remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove repeated spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()