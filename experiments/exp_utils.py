import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def check_posterior(samples: np.ndarray, gt: float, alpha: float = 0.05):
    q_low, q_high = (0 + alpha / 2.0), (1.0 - alpha / 2.0)
    ci_low, ci_high = np.quantile(samples.flatten(), [q_low, q_high])

    return (gt >= ci_low) and (gt < ci_high)


def plot_ppc(samples: np.ndarray, gt: float, alpha: float = 0.05, title:str = ''):
    """
    Plot posterior predictive check
    :param samples: posterior draws
    :param gt: ground truth param
    :param alpha: Credible Interval signifance level
    :return:
    """

    post_mean = samples.mean()
    q_low, q_high = (0 + alpha / 2.0), (1.0 - alpha / 2.0)
    ci_low, ci_high = np.quantile(samples.flatten(), [q_low, q_high])

    sns.kdeplot(samples)
    plt.axvline(gt, c='red', linestyle='--', label='True: {:.3f}'.format(gt))
    plt.axvline(post_mean, c='green', linestyle='--', label='Post. mean: {:.3f}'.format(post_mean))
    plt.axvline(ci_low, c='gray', linestyle='-.', label='Low: {:.3f}'.format(ci_low))
    plt.axvline(ci_high, c='gray', linestyle='-.', label='High: {:.3f}'.format(ci_high))
    plt.legend()
    plt.title(title)
    plt.show()
