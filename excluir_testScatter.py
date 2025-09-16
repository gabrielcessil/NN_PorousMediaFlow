import Domain_Plotter as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
np.random.seed(42)
target_data = np.random.rand(10, 10, 10) * 10
prediction_data = target_data + np.random.randn(10, 10, 10) * 2

# Call the new function with the sample data.
pl.plot_density_scatter(target_data, prediction_data)