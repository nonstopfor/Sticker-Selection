from torchmetrics import Metric
import torch

class MyAccuracy(Metric):
    def __init__(self, device="cpu", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0., device=device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0., device=device), dist_reduce_fx="sum")

    def update(self, correct, total):
        
        self.correct += correct
        self.total += total

    def compute(self):
        if self.total.item() == 0.:
            return self.total
        return self.correct.float() / self.total