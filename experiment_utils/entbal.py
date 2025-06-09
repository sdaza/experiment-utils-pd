import numpy as np
import pandas as pd
from scipy.optimize import minimize


class entbal:
    def __init__(self, n_digits=3, verbose=False):
        self.n_digits = n_digits
        self.verbose = verbose
        self.bal_tables = None

    def fit(self, X, TA, estimand="ATE", n_moments=2, base_weights=None):
        self.n_moments = n_moments
        self.X = X
        self.XD = design_matrix(self.X, self.n_moments)
        self.covnames = self.X.columns

        self.TA = TA
        self.TA_lvls = np.unique(self.TA)
        self.TA_lvls.sort()

        if base_weights is None:
            self.base_weights = np.ones(self.X.shape[0])
        else:
            self.base_weights = base_weights

        if estimand.upper() not in ["ATE", "ATC", "ATT"]:
            print("Invalid Estimand")
            return
        else:
            self.estimand = estimand

        # Setting targets
        if estimand == "ATE":
            self.tars = self.XD.mean(axis=0)
        elif estimand == "ATT":
            self.tars = self.XD[self.TA == 1].mean(axis=0)
        elif estimand == "ATC":
            self.tars = self.XD[self.TA == 0].mean(axis=0)

        def loss_fn(pars):
            return eb_loss(pars, self.XD, self.base_weights, self.tars)

        self.W = np.ones(self.X.shape[0])

        # Solving for parameters
        if estimand == "ATE":
            self.pars0 = minimize(
                lambda p: eb_loss(p, self.XD[self.TA == 0], self.base_weights[self.TA == 0], self.tars),
                x0=np.zeros(self.XD.shape[1]),
                method="l-bfgs-b",
            )
            self.W[self.TA == 0] = eb_weights(self.pars0.x, self.XD[self.TA == 0], self.base_weights[self.TA == 0])

            self.pars1 = minimize(
                lambda p: eb_loss(p, self.XD[self.TA == 1], self.base_weights[self.TA == 1], self.tars),
                x0=np.zeros(self.XD.shape[1]),
                method="l-bfgs-b",
            )
            self.W[self.TA == 1] = eb_weights(self.pars1.x, self.XD[self.TA == 1], self.base_weights[self.TA == 1])

        elif estimand == "ATT":
            self.pars0 = minimize(
                lambda p: eb_loss(p, self.XD[self.TA == 0], self.base_weights[self.TA == 0], self.tars),
                x0=np.zeros(self.XD.shape[1]),
                method="l-bfgs-b",
            )
            self.W[self.TA == 0] = eb_weights(self.pars0.x, self.XD[self.TA == 0], self.base_weights[self.TA == 0])

        elif estimand == "ATC":
            self.pars1 = minimize(
                lambda p: eb_loss(p, self.XD[self.TA == 1], self.base_weights[self.TA == 1], self.tars),
                x0=np.zeros(self.XD.shape[1]),
                method="l-bfgs-b",
            )
            self.W[self.TA == 1] = eb_weights(self.pars1.x, self.XD[self.TA == 1], self.base_weights[self.TA == 1])

    def evaluate_balance(self, n_digits=None):
        if n_digits is None:
            n_digits = self.n_digits

        bar_xt = self.X[self.TA == 1].mean()
        bar_xc = self.X[self.TA == 0].mean()
        sd_xt = self.X[self.TA == 1].std()
        sd_xc = self.X[self.TA == 0].std()

        w_bar_xt = self.X[self.TA == 1].apply(lambda z: w_mean(self.W[self.TA == 1], z))
        w_bar_xc = self.X[self.TA == 0].apply(lambda z: w_mean(self.W[self.TA == 0], z))
        w_sd_xt = self.X[self.TA == 1].apply(lambda z: w_var(self.W[self.TA == 1], z))
        w_sd_xc = self.X[self.TA == 0].apply(lambda z: w_var(self.W[self.TA == 0], z))

        sdm0 = self.X.apply(lambda z: std_diff_means(z, self.TA))
        sdm1 = self.X.apply(lambda z: std_diff_means(z, self.TA, self.W))

        lrsd0 = self.X.apply(lambda z: log_ratio_sd(z, self.TA))
        lrsd1 = self.X.apply(lambda z: log_ratio_sd(z, self.TA, self.W))

        self.N0 = self.X[self.TA == 0].shape[0]
        self.N1 = self.X[self.TA == 1].shape[0]
        self.ESS0 = ess(self.W[self.TA == 0])
        self.ESS1 = ess(self.W[self.TA == 1])

        self.bal_tables = {
            "unweighted": pd.DataFrame(
                {
                    "Mean_XT": bar_xt,
                    "SD_XT": sd_xt,
                    "Mean_XC": bar_xc,
                    "SD_XC": sd_xc,
                    "SDM": sdm0,
                    "LRSD": lrsd0,
                }
            ),
            "weighted": pd.DataFrame(
                {
                    "Mean_XT": w_bar_xt,
                    "SD_XT": w_sd_xt,
                    "Mean_XC": w_bar_xc,
                    "SD_XC": w_sd_xc,
                    "SDM": sdm1,
                    "LRSD": lrsd1,
                }
            ),
            "treated_n": [self.N1, self.ESS1],
            "control_n": [self.N0, self.ESS0],
        }

        self.bal_res = pd.DataFrame(
            {
                "Mean_XT": bar_xt,
                "SD_XT": sd_xt,
                "Mean_XC": bar_xc,
                "SD_XC": sd_xc,
                "W_Mean_XT": w_bar_xt,
                "W_SD_XT": w_sd_xt,
                "W_Mean_XC": w_bar_xc,
                "W_SD_XC": w_sd_xc,
                "SDM": sdm0,
                "W_SDM": sdm1,
                "LRSD": lrsd0,
                "W_LRSD": lrsd1,
            }
        )

        return self.bal_res.round(n_digits)

    def print_balance(self, n_digits=3):
        if self.bal_tables is None:
            self.evaluate_balance()

        bt = self.bal_tables

        col_names = ["VarName", "barXT", "sdXT", "barXC", "sdXC", "SDM", "LRSD"]

        # pulling out variable names
        var_names = bt["unweighted"].index
        vnl = np.max(np.array([len("VarName"), *[len(vn) for vn in var_names]]))

        # building the strings
        var_block = "{:<" + str(vnl + 1) + "}|"
        num_block = " {:>6} |"
        line_str = var_block + 6 * num_block

        dash_len = len(line_str.format(*col_names))
        line_break = dash_len * "-"

        # Printing the tables
        print(line_break)
        print("Unweighted Summary Statistics")
        print(line_break)
        print(line_str.format(*col_names))
        print(line_break)
        for i, vn in enumerate(var_names):
            print(line_str.format(vn, *bt["unweighted"].iloc[i].round(n_digits)))
        print(line_break)

        print("Weighted Summary Statistics")
        print(line_break)
        for i, vn in enumerate(var_names):
            print(line_str.format(vn, *bt["weighted"].iloc[i].round(n_digits)))
        print(line_break)

        # Effective Sample Size Summaries
        ess0_string = "Control: N = {:<1}, ESS = {:<10}".format(*np.array(bt["control_n"]).round(n_digits))
        ess1_string = "Treated: N = {:<1}, ESS = {:<10}".format(*np.array(bt["treated_n"]).round(n_digits))

        print("Sample Size Summaries")
        print(line_break)
        print(ess0_string)
        print(ess1_string)
        print(line_break)


def eb_weights(pars, XD, bw):
    XDP = -XD.dot(pars)
    max_XDP = np.max(XDP)
    Q = bw * np.exp(XDP - max_XDP)
    return Q / np.sum(Q)


def eb_loss(pars, XD, bw, tars):
    XDP = -XD.dot(pars)
    max_XDP = np.max(XDP)
    return max_XDP + np.log(bw.dot(np.exp(XDP - max_XDP))) + tars.dot(pars)


def design_matrix(X, n_mom=3):
    D = [X.apply(lambda x: (x**p - np.mean(x**p)) / np.std(x**p), axis=0) for p in np.arange(1, n_mom + 1)]  # noqa: B023
    XD = pd.concat(D, axis=1)
    return XD


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def w_mean(w, x):
    return x.dot(w) / np.sum(w)


def w_var(w, x):
    z = (x - w_mean(w, x)) ** 2
    return z.dot(w) / np.sum(w)


def std_diff_means(x, ta, w=None):
    if w is None:
        w = np.ones(x.shape[0])
    bar_xt = w_mean(w[ta == 1], x[ta == 1])
    bar_xc = w_mean(w[ta == 0], x[ta == 0])
    var_xt = w_var(w[ta == 1], x[ta == 1])
    var_xc = w_var(w[ta == 0], x[ta == 0])
    sdm = (bar_xt - bar_xc) / np.sqrt((var_xt + var_xc) / 2.0)
    return sdm


def log_ratio_sd(x, ta, w=None):
    if w is None:
        w = np.ones(x.shape[0])
    sd_xt = np.sqrt(w_var(w[ta == 1], x[ta == 1]))
    sd_xc = np.sqrt(w_var(w[ta == 0], x[ta == 0]))
    return np.log(sd_xt) - np.log(sd_xc)


def ess(w):
    return np.sum(w) ** 2 / np.sum(w**2)
