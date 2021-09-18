
"""
Use this class as a base class for any
"""
class BaseModel:
    def __init__(self):
        self.ntrial = 10

    def train(self):
        return NotImplementedError

    def detect(self):
        return NotImplementedError

    def train_and_test(self):
        return NotImplementedError

    def optimize_param(self, train):
        return NotImplementedError
