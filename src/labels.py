"""Label mapping for the Sign Language MNIST dataset.

Labels are 0-25 corresponding to letters A-Z, but J (9) and Z (25) are
excluded because those two letters require motion and cannot be represented
by a single static hand pose.
"""

LABEL_TO_LETTER = {i: chr(ord("A") + i) for i in range(26) if i not in (9, 25)}


def label_to_letter(label: int) -> str:
    return LABEL_TO_LETTER[int(label)]
