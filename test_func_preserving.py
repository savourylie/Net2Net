import torch
import torch.nn as nn
import torch.nn.functional as F
from net2net import make_wider


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):
        print("Data shape: {}".format(x.size()))

        # Layer 1 (conv)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Layer 2 (conv)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Flatten
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        # FC 1
        x = self.fc1(x)
        x = F.relu(x)

        # FC 2
        x = self.fc2(x)
        x = F.relu(x)

        # FC 3
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def test_conv2conv_wider()
    x_data = torch.rand((64, 1, 28, 28))

    # Teacher model
    model = Net()
    model.eval()
    teacher_result = model(x_data).detach().numpy()

    # Student model
    model_wider = deepcopy(model)
    layer1 = model_wider.conv1
    layer2 = model_wider.conv2

    old_width = conv1.weight.data[0]
    new_width = np.random.randint(old_width + 1, old_width * 100)

    _ = model_wider(model_wider.conv1, model_wider.conv2, new_width)
    model_wider.eval()
    student_result = model_wider(x_data).detach().numpy()
    
    assert np.allclose(teacher_result, student_result)



