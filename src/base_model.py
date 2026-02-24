# Authors: Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Protocol definition for super-resolution model interface

from typing import Protocol

import numpy as np  # type: ignore
from jaxtyping import Float  # type: ignore


class BaseModel(Protocol):
    def predict(
        self, image: Float[np.ndarray, "w h c"]
    ) -> tuple[Float[np.ndarray, "w2 h2 c"], Float[np.ndarray, "w2 h2"]]:
        pass
