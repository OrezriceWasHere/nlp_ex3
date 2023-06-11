import itertools

import random

positive_file = "pos_examples"
negative_file = "neg_examples"

NUM_OF_POS_SEQ = 5000
NUM_OF_NEG_SEQ = 5000


def is_prime(num):
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True


def generate_good_examples():
    buffer = []
    while len(buffer) < NUM_OF_POS_SEQ:
        r = random.randint(100000, 10000000)
        if r not in buffer:
            if is_prime(r):
                buffer.append(bin(r)[2:])

    return buffer


def generate_bad_examples():
    buffer = []
    while len(buffer) < NUM_OF_POS_SEQ:
        r = random.randint(100000, 10000000)
        if r not in buffer:
            if not is_prime(r) and r % 2 == 1:
                buffer.append(bin(r)[2:])

    return buffer


if __name__ == "__main__":
    with open(positive_file, "w") as f:
        for example in generate_good_examples():
            f.write(example + "\n")

    with open(negative_file, "w") as f:
        for example in generate_bad_examples():
            f.write(example + "\n")
