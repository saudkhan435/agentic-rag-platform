from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 120
) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_text(text)