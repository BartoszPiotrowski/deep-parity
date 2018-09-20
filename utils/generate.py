import argparse
from random import choice


NUMBERS = [0, 1, 2]
OPERATIONS = ['+', '-']
LENGTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11]
MODULO = 2

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
    args = parser.parse_args()

for i in range(args.n):
    expression = generate_aritmetic_expression(NUMBERS, OPERATIONS, LENGTHS)
    print(modulo(expression, MODULO), expression)


