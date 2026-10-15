def chunk_text(text: str, chunk_size: int = 25, overlap: int = 5) -> list[str]:
    """Split text into sliding window chunks by word count."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_documents(documents: list[str], chunk_size: int = 25, overlap: int = 5) -> list[str]:
    """Chunks a list of whole documents."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size, overlap))
    return all_chunks