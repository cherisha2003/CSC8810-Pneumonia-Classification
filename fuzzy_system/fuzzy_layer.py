import numpy as np, matplotlib.pyplot as plt

def gaussian(x,m,s):
    return np.exp(-((x-m)**2)/(2*s**2+1e-6))

def fuzzy_transform(X,m,s):
    return gaussian(X,m,s)

def plot_mf(m,s):
    x=np.linspace(-3,3,100)
    plt.figure()
    for i in range(6):
        plt.plot(x,gaussian(x,m[i],s[i]))
    plt.savefig("results/fuzzy_membership_functions.png"); plt.close()

def compare_mf(m1,s1,m2,s2):
    x=np.linspace(-3,3,100)
    plt.figure()
    for i in range(3):
        plt.plot(x,gaussian(x,m1[i],s1[i]),'--')
        plt.plot(x,gaussian(x,m2[i],s2[i]))
    plt.savefig("results/fuzzy_before_after_ga.png"); plt.close()

def feature_dist(a,b):
    plt.hist(a.flatten(),50,alpha=0.5)
    plt.hist(b.flatten(),50,alpha=0.5)
    plt.savefig("results/feature_distribution.png"); plt.close()

def heatmap(X):
    plt.imshow(X[:50],aspect='auto'); plt.colorbar()
    plt.savefig("results/fuzzy_feature_heatmap.png"); plt.close()