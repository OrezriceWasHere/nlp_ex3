import numpy as np

positive_file = "pos_examples"
negative_file = "neg_examples"

NUM_OF_POS_SEQ = 500
NUM_OF_NEG_SEQ = 500

LENGTH_OF_POS_SEQ = 30


def generate_good_example() -> str:
    possible_chars = list("abcdefghijklmnopqrstuvwxyz")
    length = np.random.randint(3, LENGTH_OF_POS_SEQ)
    chars = [np.random.choice(possible_chars) for _ in range(length)]
    chars[2] = "z"
    return "".join(chars)


def generate_bad_example() -> str:
    possible_chars = list("abcdefghijklmnopqrstuvwxyz")
    length = np.random.randint(3, LENGTH_OF_POS_SEQ)
    chars = [np.random.choice(possible_chars) for _ in range(length)]
    chars_without_z = [char for char in chars if char != "z"]
    if chars[2] == "z":
        chars[2] = np.random.choice(chars_without_z)
    return "".join(chars)


if __name__ == "__main__":
    with open(positive_file, "w") as f:
        for _ in range(NUM_OF_POS_SEQ):
            f.write(generate_good_example() + "\n")

    with open(negative_file, "w") as f:
        for _ in range(NUM_OF_NEG_SEQ):
            f.write(generate_bad_example() + "\n")
