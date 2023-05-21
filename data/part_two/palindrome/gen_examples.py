import numpy as np

positive_file = "pos_examples"
negative_file = "neg_examples"

NUM_OF_POS_SEQ = 500
NUM_OF_NEG_SEQ = 500

LENGTH_OF_POS_SEQ = 20


def generate_palindrome() -> str:
    length = np.random.randint(1, LENGTH_OF_POS_SEQ / 2)
    should_add_char_in_middle = np.random.choice([True, False])
    seq = [np.random.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(length)]
    combined_string = "".join(seq + seq[::-1])
    if should_add_char_in_middle:
        random_char = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"))
        combined_string = combined_string[:len(combined_string) // 2] + \
                          random_char + \
                          combined_string[len(combined_string) // 2:]
    return combined_string


def is_palindrome(seq: str) -> bool:
    return seq == seq[::-1]


def generate_non_palindrome() -> str:
    length = np.random.randint(1, LENGTH_OF_POS_SEQ)
    seq = [np.random.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(length)]
    if is_palindrome(seq):
        seq = generate_non_palindrome()
    return "".join(seq)


if __name__ == "__main__":
    with open(positive_file, "w") as f:
        for _ in range(NUM_OF_POS_SEQ):
            f.write(generate_palindrome() + "\n")

    with open(negative_file, "w") as f:
        for _ in range(NUM_OF_NEG_SEQ):
            f.write(generate_non_palindrome() + "\n")
