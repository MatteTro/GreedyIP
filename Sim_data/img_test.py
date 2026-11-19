import numpy as np

# -----------------------------
# TEST FUNCTIONS
# -----------------------------

def gaussian_mixture_signed(X, Y):
    return (
        1.0 * np.exp(
            -((X - 0.25) ** 2) / 0.01
            -((Y - 0.30) ** 2) / 0.02
        )
        + 0.8 * np.exp(
            -((X - 0.70) ** 2) / 0.03
            -((Y - 0.75) ** 2) / 0.01
        )
        + 0.6 * np.exp(
            -((X - 0.55) ** 2) / 0.015
            -((Y - 0.20) ** 2) / 0.015
        )
        - 0.3 * np.exp(
            -((X - 0.45) ** 2) / 0.02
            -((Y - 0.55) ** 2) / 0.02
        )
    ) 

# -----------------------------
# REGISTRY
# -----------------------------

FUNCTIONS = {
    "gaussian_mixture_signed": gaussian_mixture_signed
}


# -----------------------------
# ACCESSOR COMODO
# -----------------------------

def get_function(name):
    if name not in FUNCTIONS:
        raise ValueError(f"Function '{name}' not found. Available: {list(FUNCTIONS.keys())}")
    return FUNCTIONS[name]