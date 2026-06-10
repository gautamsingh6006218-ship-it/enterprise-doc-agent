import sys
from pathlib import Path
from agent.ingestion.pdf_loader import load_pdf


def verify_pdf_argument():
    args = sys.argv[1:]

    if len(args) != 1:
        raise ValueError("Provide only one argument")
    
    pdf_path = Path(args[0])

    if pdf_path.exists():
        document = load_pdf(str(pdf_path))
    else:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    
    return document


if __name__ == "__main__":
    try:
        document = verify_pdf_argument()
        print("PDF loader test successful")
        print(f"Document ID: {document.id}")
        print(f"Title: {document.title}")
        print(f"Source type: {document.source_type}")
        print(f"Source path: {document.source_path}")
        print(f"Page count: {document.metadata.get('page_count')}")
        print(f"Text length: {len(document.text)}")
        print(f"Preview: {document.text[:500]}")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)
    

    