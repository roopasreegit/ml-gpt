import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        epsilon=1e-5
        rms=np.sqrt(np.mean(np.square(x)+epsilon))
        x_rms_norm=x/rms

        x_hat=gamma*x_rms_norm

        return np.round(x_hat,4).tolist()
        pass
