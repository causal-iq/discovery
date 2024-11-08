#
#   Manual throw-by-throw generation of parameter (bias) distribution in a
#   sequence of throws of a 0.3 bias coin
#

import numpy as np
import matplotlib.pyplot as plt


N = 20  # Number of flips
BIAS_HEADS = 0.3  # The bias of the coin


bias_range = np.linspace(0, 1, 101)  # The range of possible biases
prior_bias_heads = np.ones(len(bias_range)) / len(bias_range)  # Uniform prior
flip_series = (np.random.rand(N) <= BIAS_HEADS).astype(int)  # coin flips

for flip in flip_series:
    likelihood = bias_range**flip * (1-bias_range)**(1-flip)
    evidence = np.sum(likelihood * prior_bias_heads)
    prior_bias_heads = likelihood * prior_bias_heads / evidence

plt.plot(bias_range, prior_bias_heads)
plt.xlabel('Heads Bias')
plt.ylabel('P(Heads Bias)')
plt.grid()
plt.show()
