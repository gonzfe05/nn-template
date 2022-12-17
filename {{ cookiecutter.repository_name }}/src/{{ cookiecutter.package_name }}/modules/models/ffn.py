from torch import nn

class FFN(nn.Module):
    def __init__(self, num_classes: int):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(2, 2),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(2, num_classes)
        )

    def forward(self, x):
        output = self.model(x)
        return output