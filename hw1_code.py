import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
import numpy as np

##Code for Problem 1
x = np.linspace(norm.ppf(0.001),norm.ppf(0.999), 1000)
fig = plt.figure()
plt.plot(x, norm.pdf(x))
plt.suptitle("Density of Normal Distribution w/ Mean 0, Standard Deviation 1")
plt.show()


##Code for Problem 2
sample_data = np.random.normal(size=1000)
fig1 = plt.figure()
plt.hist(sample_data, bins=20)
plt.suptitle("Histogram of Randomly Sampled Normal Data")
plt.show()

def data_estimators(sample):
    """Computes the sample mean, standard deviation, skew, and kurtosis of data

    Parameters:
    sample (numpy array): Dataset input

    Returns:
    (float, float, float, float): (Mean, Standard Deviation, Skew, Kurtosis)
    """
    mean = np.mean(sample)
    sd = np.std(sample)
    skew = scipy.stats.skew(sample)
    kurt = scipy.stats.kurtosis(sample)
    return(mean, sd, skew, kurt)

est = data_estimators(sample_data)
print(est)