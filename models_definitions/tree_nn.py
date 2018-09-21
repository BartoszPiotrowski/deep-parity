import torch
import random
import argparse
import sys
sys.path.append('.')
from utils.parse import parse


class SymbolNetwork(torch.nn.Module):
    def __init__(self, symbolion_name, symbolion_arity,
                 dim_in_out=32, dim_h=32):
        super(SymbolNetwork, self).__init__()
        self.symbolion_name = symbolion_name
        self.symbolion_arity = symbolion_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(symbolion_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        )

    def forward(self, x):
        return self.model(x)


class ConstantNetwork(torch.nn.Module):
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
    def __init__(self, dim_in=32, dim_h=32, dim_out=2): # dim_out = MODULO
        super(ModuloNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


def one_hot(elem, elems):
    if isinstance(elems, int):
        assert 0 <= elem < elems
        elems = range(elems)
    else:
        assert len(set(elems)) == len(elems)
    return [1 if e ==  elem else 0 for e in elems]

def consts_to_tensors(all_consts):
    return {v: torch.tensor([one_hot(v, all_consts)], dtype=torch.float)
                        for v in all_consts}

def instanciate_modules(symbols):
    modules = {}
    for symb in symbols:
        modules[symb] = \
            SymbolNetwork(symb, symbols[symb]) \
                    if symbols[symb] else \
            ConstantNetwork(symb, symbols[symb])
    modules['CONST'] = ConstantNetwork()
    modules['MODULO'] = ModuloNetwork()
    return modules


def parameters_of_modules(modules):
    parameters = []
    for m in modules:
        parameters.extend(modules[m].parameters())
    return parameters


def tree(term, modules, consts_as_tensors):
    if len(term) > 1:
        x = torch.cat([tree(t, modules, consts_as_tensors) for t in term[1]], -1)
        return modules[term[0]](x)
    else:
        return modules['CONST'](consts_as_tensors[term[0]])


def model(term, parser, modules, consts_as_tensors):
    parsed_term = parser(term)
    return modules['MODULO'](tree(parsed_term, modules, consts_as_tensors))


def loss(outputs, targets):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, targets)


def train(inputs, labels, model, loss, optimizer, epochs=1):
    for e in range(epochs):
        ls = []
        ps = []
        for i in range(len(inputs)):
            p = model(inputs[i])
            l = loss(p, labels[i])
            ls.append(l.item())
            ps.append(p.argmax().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l_avg = sum(ls) / len(ls)
        acc = sum(ps[i] == labels[i].item() \
                  for i in range(len(labels))) / len(labels)
        print('Epoch: {}. Loss {}. Accuracy {}.'.format(e, l_avg, acc))


############ TEST ###############################################


MODULO = 2
NUMBERS = ['0', '1', '2']
SYMBOLS_WITH_ARITIES = {
    '+': 2,
    '-': 2
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_set",
    type=str,
    help="Path to a training set.")
parser.add_argument(
    "--valid_set",
    default='data/split/equiv.valid',
    type=str,
    help="Path to a validation set.")
parser.add_argument(
    "--test_set",
    default='',
    type=str,
    help="Path to a testing set.")
parser.add_argument(
    "--model_path",
    default='',
    type=str,
    help="Path where to save the trained model.")
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs.")
parser.add_argument(
    "--embed_dim",
    default=8,
    type=int,
    help="Token embedding dimension.")
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="Maximum number of threads to use.")
parser.add_argument(
    "--logdir",
    default='',
    type=str,
    help="Logdir.")
args = parser.parse_args()


consts_as_tensors = consts_to_tensors(NUMBERS)
modules = instanciate_modules(SYMBOLS_WITH_ARITIES)
params_of_modules = parameters_of_modules(modules)

labels = []
inputs = []
with open(args.train_set, 'r') as f:
    for line in f:
        label, term = line.strip('\n').split(' ')
        labels.append(torch.tensor([int(label)]))
        inputs.append(term)

model_1 = lambda term: model(term, parse, modules, consts_as_tensors)
loss_1 = loss
optimizer_1 = torch.optim.SGD(params_of_modules, lr=0.001, momentum=0.9)
train(inputs, labels, model_1, loss_1, optimizer_1, epochs=args.epochs)


############ TEST ENDED #########################################

