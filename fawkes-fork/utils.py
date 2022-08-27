from string import ascii_letters
from random import randint


class Utils:

    @staticmethod
    def random_str(letters=6):
        return "".join([
            ascii_letters[
                randint(0, len(ascii_letters))]
            for _ in
            range(letters)
        ])
 