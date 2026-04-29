class VidSmootherError(RuntimeError):
    """Base error for expected pipeline failures."""


class ToolMissingError(VidSmootherError):
    """Raised when a required executable or model path is unavailable."""


class CommandError(VidSmootherError):
    """Raised when an external command exits unsuccessfully."""
