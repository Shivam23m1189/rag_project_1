import glob

def load_documents(directory: str) -> list[str]:
    """Reads all .txt files from the target directory and subdirectories."""
    documents = []
    for filepath in glob.glob(f"{directory}/**/*.txt", recursive=True):
        with open(filepath, "r", encoding="utf-8") as file:
            documents.append(file.read())
    return documents