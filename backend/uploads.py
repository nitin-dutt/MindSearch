"""Safe handling of uploaded files (alignment_plan SEC-04).

The client-supplied filename is never used as a path. Files are stored under a random name with an
allow-listed extension, checked for size and for a matching file signature.
"""
import os
import uuid

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

# first bytes of real files of each binary type (a renamed .exe must not pass as a .pdf)
_SIGNATURES = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}


class UploadError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def safe_upload_path(upload_dir: str, filename: str) -> str:
    """Return upload_dir/<random>.<ext>. The original name only contributes its (validated) extension."""
    ext = os.path.splitext(os.path.basename((filename or "").replace("\\", "/")))[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"File type '{ext or '(none)'}' is not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")


def validate_content(path: str, data: bytes) -> None:
    if len(data) == 0:
        raise UploadError("The file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", status_code=413)
    signature = _SIGNATURES.get(os.path.splitext(path)[1])
    if signature and not data.startswith(signature):
        raise UploadError("The file content does not match its extension")
