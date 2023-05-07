import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import scipy.stats as st


def calibration_regression(mean, sigma, Y,label, color, ax=None):
    """
    mean : (n_samples,1) or (n_sample,) prediction mean 
    sigma : (n_samples,1) or (n_sample,) prediction sigma 
    Y : (n_samples,1) or (n_sample,) Y co-ordinate of ground truth 
    label :  string, 
    
    
    """

    marker_size = None
    if ax is None:
        fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["mean"] = mean
    df["sigma"] = sigma
    df["Y"] = Y
    df["z"] = (df["Y"] - df["mean"]) / df["sigma"]
    df["perc"] = st.norm.cdf(df["z"])
    k = jnp.arange(0, 1.1, 0.1)
    counts = []
    df2 = pd.DataFrame()
    df2["Interval"] = k
    df2["Ideal"] = k
    for i in range(0, 11):
        l = df[df["perc"] < 0.5 + i * 0.05]
        l = l[l["perc"] >= 0.5 - i * 0.05]
        counts.append(len(l) / len(df))
    df2["Counts"] = counts

    ax.plot(k, counts, color=color, label=label)

    ax.scatter(k, counts, color=color,s=marker_size)
    ax.scatter(k, k,color="green",s=marker_size)
    ax.set_yticks(k)
    ax.set_xticks(k)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.legend()
    ax.set_xlabel("decile")
    ax.set_ylabel("ratio of points")
    ax.plot(k, k, color="green")
    sns.despine()
    return df, df2