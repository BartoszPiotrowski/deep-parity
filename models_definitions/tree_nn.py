import random
import torch
import sys
sys.path.append('.')
from utils.parse import parse


class FunctionNetwork(torch.nn.Module):
    def __init__(self, function_name, function_arity,
                 dim_in_out=32, dim_h=32):
        super(FunctionNetwork, self).__init__()
        self.function_name = function_name
        self.function_arity = function_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(function_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        )

    def forward(self, x):
        return self.model(x)


class ConstantNetwork(torch.nn.Module):
    # TODO bias=True?
    def __init__(self, dim_in=3, dim_h=32, dim_out=32):
        super(ConstantNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)

class ModuloNetwork(torch.nn.Module):
    # TODO bias=True?
    def __init__(self, dim_in=3, dim_h=32, dim_out=1): # or dim_out = MODULO
        super(ModuloNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


def loss(y_pred, y):
    return (y_pred - y).pow(2).item() # TODO not pow(2) but |.|


def one_hot(elem, elems):
    if isinstance(elems, int):
        assert 0 <= elem < elems
        elems = range(elems)
    else:
        assert len(set(elems)) == len(elems)
    return [1 if e ==  elem else 0 for e in elems]

def constants_as_tensors(all_constants):
    return {v: torch.tensor([one_hot(v, all_constants)], dtype=torch.float)
                        for v in all_constants}

############ TEST ###############################################
MODULO = 2
NUMBERS = ['0', '1', '2']
FUNCTS_WITH_ARITIES = {
    '+': 2,
    '-': 2
}

constants = constants_as_tensors(NUMBERS)

term = '1-2+1'
term_2 = '2+2+0-0-1+0-2-1+0-2-1-2'


parsed_term = parse(term)
parsed_term_2 = parse(term_2)
print(parsed_term)
print(parsed_term_2)

functs = FUNCTS_WITH_ARITIES
components = {} # Dictionary for instances of FunctionNetwork and ConstantNetwork
for func in functs:
    components[func] = \
        FunctionNetwork(func, functs[func]) \
                if functs[func] else \
        ConstantNetwork(func, functs[func])
components['CONST'] = ConstantNetwork()

def tree(term):
    if len(term) > 1:
        x = torch.cat([tree(arg) for arg in term[1]], -1)
        return components[term[0]](x)
    else:
        assert term[0] in NUMBERS
        return components['CONST'](constants[term[0]])

print(tree(parsed_term))
print(tree(parsed_term_2))
#print(eq(torch.cat((tree(parsed_term), tree(parsed_term_2)), -1)))
#l = loss(1, eq(torch.cat((tree(parsed_term), tree(parsed_term_2)), -1)))
#print(l)

sys.exit()

############ TEST ENDED #########################################


class TermNN(torch.nn.Module):
    def __init__(self, functs_with_arities, variables):
        """
        Define and instantiate small component networks for each function.
        """
        super(TermNN, self).__init__()

        self.components = {} # Dictionary for instances of FunctionNetwork
        for pred in functs_with_arities:
            self.components[pred] = FunctionNetwork(
                pred,
                functs_with_arities[pred]
            )

    def tree(self, term):
        if term[1]:
            x = torch.cat([tree(c) for c in term[1]], -1)
            return self.components[term[0]](x)
        else:
            return self.components['variable'](term[0])
            # TODO case of constant e

    def forward(self, term):
        """
        For given term compute forward pass in tree-like network which shape
        corresponds to the shape of the term. Term is assumed to be of the parsed
        form [function, list_of_arguments].
        """
        y_pred = tree(term)
        return y_pred



#####################################

DIM_IN_OUT, DIM_H = 32, 32


# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# TODO remove it
test_example = 'k(X,t(t(b(o(X,Y),Z),o(Y,X)),U))'

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

