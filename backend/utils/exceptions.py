class RecommenderError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int = 500, details: dict | None = None):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class InvalidProfileError(RecommenderError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("INVALID_PROFILE", message, 422, details)


class NoSafeCandidateError(RecommenderError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("NO_SAFE_CANDIDATE", message, 409, details)


class NoFeasiblePlanError(RecommenderError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("NO_FEASIBLE_PLAN", message, 409, details)


class ModelNotReadyError(RecommenderError):
    def __init__(self, message: str = "Model artifacts not loaded"):
        super().__init__("MODEL_NOT_READY", message, 503)
