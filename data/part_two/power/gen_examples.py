import itertools

import numpy as np

positive_file = "pos_examples"
negative_file = "neg_examples"

NUM_OF_POS_SEQ = 500
NUM_OF_NEG_SEQ = 500


def generate_positive_string_buffer() -> list[str]:
    """
    In this language, the number of bs in the string must be a power of 2 to the numbers of as.
    In order to generate the language, we choose the number of a. Let's say 2.
    So there has to be 4 bs in the string.
    To build this string, we choose the location of the two a in 5 slot,
    between each b, after the last or before the first.
    So when there is 2a, there are 5c2 = 10 strings.

    We need 500 examples at all.
    for #(a) = 1 -> 2c1 options.
    for #(a) = 2 -> 5c2 = 10 options.
    for #(a) = 3 -> 10c3 = 120
    for #(a) = 4 -> 17c4 = 2380.
    That should be enough to choose from.
    """

    buffer = list()

    for number_of_a in range(1, 5):
        # We need to find all possible indexes to place a.
        number_of_b = number_of_a ** 2
        indexes_to_place_a = list(range(number_of_b + 1))
        # now we need to find all permutation in length (number of a) of the indexes.
        # We can use itertools.permutations for that.
        permutations = [tuple(sorted(order)) for order in itertools.permutations(indexes_to_place_a, number_of_a)]
        permutations = set(permutations)
        # Now we need to find all possible combinations of the permutations with the bs.
        # We can use itertools.product for that.
        for permutation in permutations:
            sequence = list("b" * number_of_b)
            for offset, index in enumerate(permutation):
                sequence.insert(index + offset, "a")
            result = "".join(sequence)
            buffer.append(result)
    return buffer


def generate_good_examples(buffer) -> list[str]:
    return np.random.choice(buffer, size=NUM_OF_POS_SEQ, replace=False).tolist()


def generate_bad_examples(buffer) -> list[str]:
    bad_examples_buffer = np.random.choice(buffer, size=NUM_OF_NEG_SEQ, replace=False).tolist()
    # choose a random character from the buffer and replace it with a drop it,
    # for every string in the buffer.
    for i in range(len(bad_examples_buffer)):
        string = bad_examples_buffer[i]
        while sum([1 for char in string if char == 'a']) == sum([1 for char in string if char == 'b']) ** 2:
            count_indexes_to_replace = np.random.randint(0, len(string))
            indexes_to_replace = np.random.choice(list(range(len(string))), size=count_indexes_to_replace, replace=False)
            for index in indexes_to_replace:
                char = string[index]
                if char == "a":
                    bad_examples_buffer[i] = string[:index] + "b" + string[index + 1:]
                else:
                    bad_examples_buffer[i] = string[:index] + "a" + string[index + 1:]

    return bad_examples_buffer


if __name__ == "__main__":
    buffer = generate_positive_string_buffer()
    good_examples = generate_good_examples(buffer)
    bad_examples = generate_bad_examples(buffer)
    with open(positive_file, "w") as f:
        for example in good_examples:
            f.write(example + "\n")

    with open(negative_file, "w") as f:
        for example in bad_examples:
            f.write(example + "\n")
