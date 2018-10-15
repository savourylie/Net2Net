import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from net2net import *
from copy import deepcopy

class Net2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    # def make_wider(self, new_width):
    #     # self.conv1, self.conv2 = Net2Net._widen(self.conv1, self.conv2, new_width)
    #     # self.conv2, self.fc1 = Net2Net._widen(self.conv2, self.fc1, new_width)
    #     self.fc2, self.fc3 = Net2Net._widen(self.fc2, self.fc3, new_width)


def make_wider(layer1, layer2, new_width, batchnorm=None):
    weight1, bias1 = layer1.weight.data, layer1.bias.data
    weight2 = layer2.weight.data

    layer1_num_input_units, layer1_num_output_units = weight1.size(1), weight1.size(0)
    layer2_num_input_units, layer2_num_output_units = weight2.size(1), weight2.size(0)

    old_width = layer1_num_output_units

    g_mapping = {j: j if j < old_width else np.random.choice(old_width) for j in range(new_width)}
    g_counter = Counter(g_mapping.values())

    ## Check if the new width is wider
    if layer1_num_output_units >= new_width:        
        raise ValueError("New width has to be larger than current width.")

    weight1_widened, bias1_widened = weight1.clone(), bias1.clone()
    weight2_widened = weight2.clone()

    if "Conv" in layer1.__class__.__name__: # When current layer is conv
        # Check current weight rank
        assert weight1.dim() == 4
        w1_kh_size, w1_kw_size = weight1.size(2), weight1.size(3)
        
        # Widen weight1/bias1
        weight1_widened.resize_(new_width, layer1_num_input_units, w1_kh_size, w1_kw_size)
        bias1_widened.resize_(new_width)

        # Intialize the extended parts with zeros
        weight1_widened[old_width:new_width, :, :, :] = 0

        for j in range(old_width, new_width):
            weight1_widened[j, :, :, :] = weight1[g_mapping[j], :, :, :]
            bias1_widened[j] = bias1[g_mapping[j]]

        # Update weights for layer 1
        layer1.weight.data, layer1.bias.data = weight1_widened, bias1_widened

        if "Conv" in layer2.__class__.__name__: # When next layer is conv
            ## Check if the output channels of layer 1 matches the input channels of layer 2
            assert layer1_num_output_units == layer2_num_input_units
            # Check next weight rank
            assert weight2.dim() == 4
            w2_kh_size, w2_kw_size = weight2.size(2), weight2.size(3)

            # Widen weight2
            weight2_widened.resize_(layer2_num_output_units, new_width, w2_kh_size, w2_kw_size)
            # Intialize the extended parts with zeros
            weight2_widened[:, old_width:new_width, :, :] = 0

            for j in range(new_width):
                weight2_widened[:, j, :, :] = weight2[:, g_mapping[j], :, :] / g_counter[g_mapping[j]]

            # Update weights for layer 2
            layer2.weight.data = weight2_widened

            return layer1, layer2

        elif "Linear" in layer2.__class__.__name__: # When next layer is linear
            # Check next weight rank
            assert weight2.dim() == 2

            # Get feature map size and side length
            feature_map_size = layer2_num_input_units // layer1_num_output_units
            feature_map_side_length = int(np.sqrt(feature_map_size))

            weight2_reshaped = weight2_widened.reshape(layer2_num_output_units, layer1_num_output_units, \
                              feature_map_side_length, feature_map_side_length) 
            weight2_widened = weight2_widened.reshape(layer2_num_output_units, layer1_num_output_units, \
                              feature_map_side_length, feature_map_side_length).clone()

            # Widen weight2
            weight2_widened.resize_(layer2_num_output_units, new_width, feature_map_side_length, feature_map_side_length)

            # Intialize the extended parts with zeros
            weight2_widened[:, old_width:new_width, :, :] = 0

            for j in range(new_width):
                weight2_widened[:, j, :, :] = weight2_reshaped[:, g_mapping[j], :, :] / g_counter[g_mapping[j]]

            # Reshape weight 2 back to the flattened form
            new_width_fc = new_width * feature_map_side_length ** 2
            weight2_widened = weight2_widened.view(layer2_num_output_units, new_width_fc)

            # Update weights for layer 2
            layer2.weight.data = weight2_widened
            layer2.in_features = new_width_fc

            return layer1, layer2

        else:
            raise TypeError("Layer 2 has to be either convolutional or linear.")

    elif "Linear" in layer1.__class__.__name__: # When layer 1/layer 2 are both linear 
        assert weight1.dim() == 2 and weight2.dim() == 2

        # Widen weight1/bias1
        weight1_widened.resize_(new_width, layer1_num_input_units)
        bias1_widened.resize_(new_width)
        weight2_widened.resize_(layer2_num_output_units, new_width)

        # Intialize the extended parts with zeros
        weight1_widened[old_width:new_width, :] = 0
        weight2_widened[:, old_width:new_width] = 0

        for j in range(old_width, new_width):
            weight1_widened[j, :] = weight1[g_mapping[j], :]
            bias1_widened[j] = bias1[g_mapping[j]]

        for j in range(new_width):
            weight2_widened[:, j] = weight2[:, g_mapping[j]] / g_counter[g_mapping[j]]
        
        # Update weights for layer 1/layer 2
        layer1.weight.data, layer1.bias.data = weight1_widened, bias1_widened
        layer1.out_features = new_width

        layer2.weight.data = weight2_widened
        layer2.in_features = new_width

        return layer1, layer2

    else:
        raise ValueError("Layer type not supported.")


if __name__ == '__main__':
    x_data = torch.rand((64, 1, 28, 28))

    model = Net2Net()
    print("\nTeacherNet architecture:")
    print(model)
    print("")
    model.eval()
    teacher_result = model(x_data).detach().numpy()
    # print(teacher_result)
    # print(teacher_result.shape)

    model_wider = deepcopy(model)
    model_wider.make_wider(1000)
    print("StudentNet architecture:")
    print(model_wider)
    print("")
    model_wider.eval()
    student_result = model_wider(x_data).detach().numpy()
    # print(student_result)
    # print(student_result.shape)
    print(np.allclose(teacher_result, student_result))