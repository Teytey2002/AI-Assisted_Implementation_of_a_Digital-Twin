from typing import Dict

class Config:
    def __init__(self, config_id: int, params: Dict):
        self.id = config_id
        self.params = params
        self.history = []  # val_loss per iteration

    def add_result(self, val_loss: float):
        self.history.append(val_loss)

    def __repr__(self):
        return f"Config(id={self.id}, params={self.params})"