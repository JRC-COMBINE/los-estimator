# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from los_estimator.fitting.distributions import Distributions


# %%
noise_factor = 1


def generate_curve(length=1000, random_seed=42):
    np.random.seed(random_seed)
    r = np.random.normal(0, noise_factor, size=length)
    r = np.zeros(length)
    x = np.arange(0, length)
    y = np.zeros_like(x, dtype=float)
    y += np.sin((x + r) / 70)
    y += (-(((x + r) / 1000 - 0.5) ** 2)) * 10
    y += np.random.normal(0, 0.1, size=len(x)) * noise_factor
    y += -y.min()
    y *= 60
    return y


def generate_and_save_synthetic_data(plot=False, noise_factor=1, length=1000, kernel_width=100, random_seed=42):
    np.random.seed(random_seed)
    admissions = generate_curve(length)
    admissions = np.concatenate([np.zeros(100), generate_curve(length)])
    los = Distributions.generate_kernel(
        "lognorm",
        [1.2, 0.7, 0.1],
        kernel_size=kernel_width,
    )
    kernel = 1 - los.cumsum()

    occupancy = np.zeros(length + len(kernel))
    for t in range(1, length):
        for t2 in range(len(kernel)):
            occupancy[t + t2] += admissions[t] * kernel[t2] + np.random.normal(0, 0.01) * noise_factor
    if plot:
        plt.plot(admissions * 10, label="Admissions")
        plt.plot(occupancy, label="Occupancy")
        plt.title("Synthetic ICU Data")
        plt.legend()
        plt.show()

    dates = pd.date_range(start="2020-01-01", periods=len(admissions), freq="D")
    data = {
        "icu_admissions": admissions.astype(float),
        "icu_occupancy": occupancy.astype(float),
    }

    df = pd.DataFrame(data, index=dates)
    data_path = "synthetic_icu_data.csv"
    df.to_csv(data_path)

    df_original_kernel = pd.DataFrame(
        {"los": los},
        index=np.arange(len(los)),
    )
    kernel_path = "synthetic_icu_kernel.csv"
    df_original_kernel.to_csv(kernel_path)
    return kernel_width, data_path, kernel_path
