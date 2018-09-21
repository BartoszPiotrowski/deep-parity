import argparse
from random import choice



def generate_aritmetic_expression(numbers, operations, lengths):
    expression = ''
    l = choice(lengths)
    for j in range(l):
        expression += str(choice(numbers))
        expression += choice(operations)
    expression += str(choice(numbers))
    return expression

def modulo(expression, modulo=2):
    return eval(expression) % modulo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", default=32, type=int, help="Number of examples to generate.")
    parser.add_argument(
        "--modulo", default=2, type=int, help="Dividing modulo this number.")
    parser.add_argument(
        "--numbers",
        type=str,
        default='0,1,2',
        help="List of numbers for composing expressions. Delimited by ','."
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default='3,4,5,6,7,8,9,10,11',
        help="List of possible lengths of expressions. Delimited by ','."
    )
    parser.add_argument(
        "--operations",
        type=str,
        default='-,+',
        help="List of arithmetical operations for composing expressions. \
        Delimited by ','."
    )
    args = parser.parse_args()

numbers = [int(n) for n in args.numbers.split(',')]
lengths = [int(n) for n in args.lengths.split(',')]
operations = args.operations.split(',')

for i in range(args.n):
    expression = generate_aritmetic_expression(numbers, operations, lengths)
    print(modulo(expression, args.modulo), expression)


