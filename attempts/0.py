import random
import torch
import sys
sys.path.append('.')
from utils.parse import parse

torch.manual_seed(541)
random.seed(541)


class Network(torch.nn.Module):
    def __init__(self, dim_in=4, dim_h=4, dim_out=4):
        super(Network, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


def loss(outputs, targets):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, targets)


############ TEST ###############################################
N_EXAMPLES = 10
DIM_EXAMPLES = 5
N_TARGETS = 4

inputs = torch.rand(N_EXAMPLES,DIM_EXAMPLES)
targets = torch.tensor([random.randint(0,N_TARGETS-1) for _ in range(N_EXAMPLES)])

components = {} # Dictionary for instances of FunctionNetwork and ConstantNetwork
parameters_list = []

components[0] = Network(dim_in=DIM_EXAMPLES, dim_out=4)
components[1] = Network(dim_in=4, dim_out=4)
components[2] = Network(dim_in=4, dim_out=N_TARGETS)

parameters_list.extend(components[0].parameters())
parameters_list.extend(components[1].parameters())
parameters_list.extend(components[2].parameters())

print(components[0].parameters())
print(parameters_list)

def model(x):
    return components[2](
        components[1](
        components[0](x)))

l = loss(model(inputs), targets)
print(l)

w = list(components[1].parameters())[2]
print('Before:'); print(w); print(w.grad)

optimizer = torch.optim.SGD(parameters_list, lr=0.001, momentum=0.9)
optimizer.zero_grad()
l.backward()
optimizer.step()

print('After:'); print(w); print(w.grad)

sys.exit()


p_m = components['MODULO'].parameters()
p_plus = components['+'].parameters()
print(type(p_m))
p = p_m + p_plus
print(type(p))
for w in p:
    print(w)
sys.exit()





############ TEST ENDED #########################################

