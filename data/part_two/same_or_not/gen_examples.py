import numpy as np

positive_file = "pos_examples"
negative_file = "neg_examples"

NUM_OF_POS_SEQ = 5000
NUM_OF_NEG_SEQ = 5000

LENGTH_OF_POS_SEQ = 1000


def generate_palindrome() -> str:
    presuf = np.random.choice(['0', '1'])
    seq = ''.join([np.random.choice(list("01")) for _ in range(LENGTH_OF_POS_SEQ)])
    return presuf + seq + presuf


def generate_non_palindrome() -> str:
    presuf = np.random.choice(['0', '1'])
    seq = ''.join([np.random.choice(list("01")) for _ in range(LENGTH_OF_POS_SEQ)])
    return presuf + seq + ('1' if presuf == '0' else '0')


if __name__ == "__main__":
    with open(positive_file, "w") as f:
        for _ in range(NUM_OF_POS_SEQ):
            f.write(generate_palindrome() + "\n")

    with open(negative_file, "w") as f:
        for _ in range(NUM_OF_NEG_SEQ):
            f.write(generate_non_palindrome() + "\n")
