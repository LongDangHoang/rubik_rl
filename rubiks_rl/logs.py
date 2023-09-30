import pandas as pd
import numpy as np

from typing import Dict


class RubiksLogger:
    def __init__(self, prefix: str="train"):
        self.prefix = prefix
        self.loss_df = pd.DataFrame({
            "total_loss": [],
            "state_value_loss": [],
            "policy_loss": [],
            "weight": []
        })

    def refresh(self):
        self.__init__(prefix=self.prefix)
    
    def update(self, loss_dict: Dict[str, np.ndarray]):
        self.loss_df = pd.concat([self.loss_df, pd.DataFrame(loss_dict)], axis=0, ignore_index=True)
    
    @property
    def avg_weighted_loss(self):
        return (self.loss_df["total_loss"] * self.loss_df["weight"]).mean()

    @property
    def avg_weighted_policy_loss(self):
        return (self.loss_df["policy_loss"] * self.loss_df["weight"]).mean()

    @property
    def avg_weighted_state_value_loss(self):
        return (self.loss_df["state_value_loss"] * self.loss_df["weight"]).mean()

    @property
    def avg_total_loss_by_weight(self):
        return self.loss_df.groupby("weight")["total_loss"].mean().sort_index()
    
    @property
    def avg_state_value_loss_by_weight(self):
        return self.loss_df.groupby("weight")["state_value_loss"].mean().sort_index()

    @property
    def avg_policy_loss_by_weight(self):
        return self.loss_df.groupby("weight")["policy_loss"].mean().sort_index()
