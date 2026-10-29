class CollectorError(RuntimeError):
    """Recoverable per-platform collection error."""


class ConfigurationError(CollectorError):
    """A required credential or setting is missing."""


class AuthenticationRequired(CollectorError):
    """The platform requires a refreshed login session."""


class ChallengeDetected(CollectorError):
    """A captcha or bot challenge was detected."""

