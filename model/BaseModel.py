
"""
Use this class as a base class for any
"""
class BaseModel:
    def __init__(self):
        pass

    def train(self):
        return NotImplementedError

    def detect(self):
        return NotImplementedError