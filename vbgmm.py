# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib as mpl


class VariationalGaussianMixture():
    """Variational Bayesian estimation of a Gaussian mixture."""

    def __init__(self, N, K, D, x):
        self.N = N
        self.K = K
        self.D = D
        self.x = x
        self.alpha0 = D + 1
        self.beta0 = D + 1
        self.nu0 = D + 1
        self.u0 = self._get_u0()
        self.V0 = self._get_V0()
        self._init_params()

    def _logdet(self, mat):
        (sign, logdet) = np.linalg.slogdet(mat)
        return logdet if sign == 1 else 0

    def _get_u0(self):
        return np.mean(self.x, axis=0)

    def _get_V0(self):
        diff = self.x - self.u0
        return np.linalg.inv(self.nu0 * diff.T.dot(diff) / self.N)

    def _init_u(self):
        cov = np.linalg.inv(self.nu0*self.V0) / self.beta0
        return np.random.multivariate_normal(self.u0, cov)

    def _init_params(self):
        self.alpha = np.tile(self.alpha0, self.K)
        self.beta = np.tile(self.beta0, self.K)
        self.nu = np.tile(self.nu0, self.K)
        self.u = np.array([self._init_u() for _ in range(self.K)])
        self.V = np.tile(self.V0, (self.K, 1, 1))

        self.rho = np.zeros((self.N, self.K))
        self.r = np.zeros((self.N, self.K))
        self.eta = np.zeros(self.K)

    def _update_rho(self, n, k):
        sum_digamma = np.sum([sp.digamma(0.5*(self.nu[k]+1-d))
                              for d in range(1, self.D+1)])
        diff = self.x[n, :] - self.u[k, :]

        self.rho[n, k] = np.exp(sp.digamma(self.alpha[k]) +
                                0.5*(sum_digamma +
                                     self._logdet(self.V[k, :]) -
                                     self.D/self.beta[k] - self.nu[k] *
                                     diff.dot(self.V[k, :]).dot(diff.T)
                                     )
                                )

    def _update_r(self, n, k):
        tmpsum = np.sum(self.rho[n, :])
        self.r[n, k] = self.rho[n, k] / tmpsum if tmpsum != 0 else 0

    def _update_eta(self, k):
        self.eta[k] = np.sum(self.r[:, k])

    def _update_alpha(self, k):
        self.alpha[k] = self.alpha0 + self.eta[k]

    def _update_beta(self, k):
        self.beta[k] = self.beta0 + self.eta[k]

    def _update_nu(self, k):
        self.nu[k] = self.nu0 + self.eta[k]

    def _update_u(self, k):
        self.u[k, :] = self.beta0*self.u0 + self.r[:, k].T.dot(self.x)
        self.u[k, :] /= self.beta[k]

    def _update_V(self, k):
        cov_sample = np.zeros_like(self.V0)
        diff_x = self.x - self.u[k]
        for n in range(self.N):
            cov_sample += self.r[n, k]*np.outer(diff_x[n, :], diff_x[n, :])
        diff = self.u[k] - self.u0

        self.V[k, :] = np.linalg.inv(
            np.linalg.inv(self.V0) +
            self.beta0 * np.outer(diff, diff) +
            cov_sample
        )

    def update_params(self):
        for k in range(self.K):
            for n in range(self.N):
                self._update_rho(n, k)
                self._update_r(n, k)
            self._update_eta(k)
            self._update_alpha(k)
            self._update_beta(k)
            self._update_nu(k)
            self._update_u(k)
            self._update_V(k)

    def fit(self, eps=1e-2, max_epochs=50, print_diff=False):
        epochs = 0
        while(True):
            eta_prev = self.eta.copy()
            self.update_params()
            epochs += 1
            diff = np.linalg.norm(self.eta - eta_prev)
            if print_diff:
                print(diff)
            if (epochs > 5 and
                    (diff < eps or
                     np.abs(diff_prev - diff) < eps or
                     epochs >= max_epochs)):
                break
            diff_prev = diff
        print(f'# iterations: {epochs}')
        self.map_estimate()

    def map_estimate(self):
        self.w = np.zeros(self.K)
        self.means = self.u
        self.covs = np.zeros_like(self.V)
        for k in range(self.K):
            self.w[k] = self.alpha[k]/np.ones(self.K).dot(self.alpha)
            self.covs[k, :] = np.linalg.inv(self.nu[k] * self.V[k, :])
        print(f'w: {self.w}\nmeans: {self.means}\ncovs: {self.covs}')

    def kl_divergence(self):
        sum_kl = sp.gammaln(self.K*self.alpha0 + self.N) - \
            sp.gammaln(self.K*self.alpha0)
        tmp = self.r*np.log(self.r)
        tmp[np.isnan(tmp)] = 0
        sum_kl += np.sum(tmp)
        for k in range(self.K):
            sum_kl += (sp.gammaln(self.alpha0)-sp.gammaln(self.alpha[k]) +
                       sp.multigammaln(self.nu0*.5, self.D) -
                       sp.multigammaln(self.nu[k]*.5, self.D) +
                       0.5*(self._logdet(self.V0) * self.nu0 -
                            self._logdet(self.V[k, :])*self.nu[k] +
                       (np.log(self.beta[k])-np.log(self.beta0))*self.D))
        return sum_kl

    def _make_ellipses(self, k, ax):
        cov = self.covs[k, :]

        w, v = np.linalg.eigh(cov)
        # w: 2 eigenvalues (length), v: 2 eigenvectors (rotation)
        angle = np.degrees(np.arctan2(v[0, 1], v[0, 0]))
        height, width = 2 * np.sqrt(w)  # diameter = 2 * radius

        ell = mpl.patches.Ellipse(self.means[k, :], height, width,
                                  angle,
                                  fc='lime', lw=0, alpha=.4)
        edge = mpl.patches.Ellipse(self.means[k, :], height, width,
                                   angle,
                                   fc='none', ec='forestgreen', lw=3)
        ax.add_artist(ell)
        ax.add_artist(edge)

    def plot_with_ellipses(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)
        ax.scatter(self.x[:, 0], self.x[:, 1],
                   c='deepskyblue', edgecolor='black', alpha=.5)
        for k in range(self.K):
            self._make_ellipses(k, ax)
        ax.scatter(self.means[:, 0], self.means[:, 1],
                   c='orange', edgecolor='k',
                   marker='*', s=150)
        plt.grid()
        plt.show()


def cluster_number_selection_by_kl(x, k_range, plot=False):
    N, D = x.shape
    kls = np.empty(len(k_range))
    for k in k_range:
        print(f'======= {k} =========')
        vgm = VariationalGaussianMixture(N, k, D, x)
        vgm.fit(print_diff=False)
        kls[k-1] = vgm.kl_divergence()
        if plot:
            vgm.plot_with_ellipses()

    argmin_k = np.argmin(kls) + 1
    print(f'argmin_k: {argmin_k}, min_kl: {kls[argmin_k]}')
    return kls, argmin_k
