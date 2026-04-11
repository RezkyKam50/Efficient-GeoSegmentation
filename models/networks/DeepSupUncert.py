import torch.nn.functional as F
import torch
import torch.nn as nn

class UncertaintyWeightedDSLoss(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
 
        self.log_vars = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, outputs, targets, criterion):
 
        total_loss = 0
        for i, out in enumerate(outputs):
     
            precision = torch.exp(-self.log_vars[i])
            task_loss = criterion(out.float(), targets)
            total_loss += precision * task_loss + self.log_vars[i]
        return total_loss

    def get_weights(self):
 
        return torch.exp(-self.log_vars).detach()