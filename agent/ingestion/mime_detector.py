"""
ingestion/mime_detector.py

What problem does this solve?
- Files from email attachments, cloud storage, and API uploads often have
  wrong or missing extensions. A file named "report.pdf" might be HTML;
  a file named "document" might be a Word document. Extension-only routing
  silently misroutes these files, producing garbage text or crashes.

Why python-magic (libmagic) instead of extension-only detection?
- libmagic reads the file's actual byte signature (magic bytes), not its name.
  A PDF always starts with "%PDF". A PNG always starts with "\x89PNG\r\n".
  These signatures cannot be faked by renaming.
- Extension as a fallback: if libmagic is unavailable or returns an unknown
  type, we fall back to the extension — preserving the existing behaviour.

Why a standalone module instead of putting this in LoaderRegistry?
- Single responsibility: mime_detector.py only detects types.
  LoaderRegistry only routes. Either can be replaced independently.
- Testability: MimeTypeDetector can be unit-tested with byte buffers
  without constructing a full registry.
"""

from pathlib import Path

try:
    import magic as _magic
    _MAGIC_AVAILABLE = True
except ImportError:
    _MAGIC_AVAILABLE = False

# MIME type → canonical file extension used by LoaderRegistry
# Why not use mimetypes stdlib? It maps MIME→extension inconsistently
# across platforms and doesn't cover all Office MIME types.
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf":      "pdf",

    # Microsoft Office (OOXML)
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":   "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         "xlsx",

    # Legacy Office
    "application/msword":       "docx",
    "application/vnd.ms-excel": "xlsx",
    "application/vnd.ms-powerpoint": "pptx",

    # Text
    "text/plain":     "txt",
    "text/html":      "html",
    "text/markdown":  "markdown",
    "text/csv":       "csv",

    # Email
    "message/rfc822":           "eml",
    "application/vnd.ms-outlook": "msg",

    # Images (for OCR path)
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/tiff": "tiff",
    "image/bmp":  "bmp",
    "image/gif":  "gif",

    # Documents
    "application/rtf":               "rtf",
    "application/epub+zip":          "epub",
    "application/vnd.oasis.opendocument.text": "odt",
}


class MimeTypeDetector:
    """
    What problem does this solve?
    - Detects a file's true format from its byte content, not its filename.
      Returns a canonical extension that LoaderRegistry can route on.

    Why a class instead of a module function?
    - Allows disabling magic detection in tests (inject disabled=True) without
      patching the module. Tests that want extension-only behaviour can do so.
    - Holds the MIME→extension mapping as instance state, making it mockable.
    """

    def __init__(self, disabled: bool = False) -> None:
        """
        Args:
        - disabled: If True, always falls back to extension detection.
                    Useful in test environments or when libmagic is unavailable.
        """
        self._enabled = _MAGIC_AVAILABLE and not disabled

    def detect_extension(self, file_path: str) -> str:
        """
        What problem does this solve?
        - Returns the canonical extension for a file, using MIME detection
          first and file extension as fallback.

        Why return extension (not MIME type)?
        - LoaderRegistry keys its _loaders dict on extensions (e.g. ".pdf").
          Returning an extension keeps the detector decoupled from MIME knowledge
          at the registry level — the registry doesn't need to know about MIME.

        Detection priority:
          1. libmagic byte-signature detection → map MIME → extension
          2. File extension (.lower().lstrip('.'))
          3. "unknown" if neither produces a known extension

        Args:
        - file_path: Absolute or relative path to the file.

        Returns lowercase extension string WITHOUT the leading dot (e.g. "pdf").
        """
        path = Path(file_path)

        if self._enabled:
            try:
                mime = _magic.from_file(str(path), mime=True)
                ext = _MIME_TO_EXT.get(mime)
                if ext:
                    return ext
            except Exception:
                # libmagic can fail on unreadable files — fall through to extension
                pass

        # Fallback: use the file's own extension
        suffix = path.suffix.lower().lstrip(".")
        return suffix if suffix else "unknown"

    @property
    def is_magic_available(self) -> bool:
        """Returns True when libmagic is installed and active."""
        return self._enabled
