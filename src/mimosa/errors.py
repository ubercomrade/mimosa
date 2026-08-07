class MimosaError(Exception):
    pass


class ModelFormatError(MimosaError):
    def __init__(self, path, message):
        self.path = path
        self.message = message
        super().__init__(f"ModelFormatError: {path}: {message}")


class ModelDimensionError(MimosaError):
    def __init__(self, message):
        self.message = message
        super().__init__(f"ModelDimensionError: {message}")


class InvariantError(MimosaError):
    def __init__(self, message):
        self.message = message
        super().__init__(f"InvariantError: {message}")


class ModelInterfaceError(MimosaError):
    def __init__(self, capability, model_type, message):
        self.capability = capability
        self.model_type = model_type
        self.message = message
        super().__init__(
            f"ModelInterfaceError ({capability}, {model_type}): {message}"
        )
