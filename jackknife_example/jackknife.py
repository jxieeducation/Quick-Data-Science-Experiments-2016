import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    values = [func(x[idx!=i]) for i in range(n)]
    mean = np.sum(values) / float(len(values))
    variance = np.var(values)
    conf = (mean - 1.96 * variance, mean + 1.96 * variance)
    return conf

x = np.random.normal(0, 2, 100)

print "confidence interval of the mean is %s" % str(jackknife(x, np.mean))

