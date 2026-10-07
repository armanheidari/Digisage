import enum

class ControllerType(enum.Enum):
    TRAINING = "train"
    INFERENCE = "inference"