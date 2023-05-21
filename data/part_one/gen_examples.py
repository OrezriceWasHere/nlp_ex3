import numpy as np
import scipy.stats as ss

pos_examples_file = "pos_examples"
neg_examples_file = "neg_examples"

NUM_OF_POS_SEQ = 500
NUM_OF_NEG_SEQ = 500


def random_int_even_dist(lower, upper) -> int:
    return np.random.randint(lower, upper + 1)


def random_sequence(pattern):
    for char_range in pattern:
        count = random_int_even_dist(lower=1, upper=10)
        for _ in range(count):
            value = random_int_even_dist(*map(ord, char_range))
            yield chr(value)


def generate_sequence(pattern) -> str:
    return "".join(random_sequence(pattern))




if __name__ == "__main__":

    pos_pattern = [
        ('1', '9'),
        ('a', 'a'),
        ('1', '9'),
        ('b', 'b'),
        ('1', '9'),
        ('c', 'c'),
        ('1', '9'),
        ('d', 'd'),
        ('1', '9')
    ]

    neg_pattern = [
        ('1', '9'),
        ('a', 'a'),
        ('1', '9'),
        ('c', 'c'),
        ('1', '9'),
        ('b', 'b'),
        ('1', '9'),
        ('d', 'd'),
        ('1', '9')
    ]

    with open(pos_examples_file, "w") as f:
        for _ in range(NUM_OF_POS_SEQ):
            f.write(generate_sequence(pos_pattern) + "\n")

    with open(neg_examples_file, "w") as f:
        for _ in range(NUM_OF_NEG_SEQ):
            f.write(generate_sequence(neg_pattern) + "\n")

