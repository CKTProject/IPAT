from scipy import stats
import numpy as np

def sta_significance_test(a, b, alpha=1e-3):
    x = np.concatenate((a, b))
    k2, p = stats.normaltest(x)
    print("p = {:g}".format(p))
    if p < alpha:
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    return p



if __name__ == '__main__':
    pts = 1000
    np.random.seed(28041990)
    a = np.random.normal(0, 1, size=pts)
    b = np.random.normal(2, 1, size=pts)
    p = sta_significance_test(a, b)
    x = np.concatenate((a, b))
