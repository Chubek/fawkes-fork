from random import randint
from string import ascii_letters

import jax.numpy as jnp
import jax

class Utils:
    @staticmethod
    def random_str(letters=6):
        return "".join([
            ascii_letters[
                randint(0, len(ascii_letters))]
            for _ in
            range(letters)
        ])





