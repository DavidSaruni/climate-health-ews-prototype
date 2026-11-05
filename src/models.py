"""Predictive models (proposal section 5.7.4).

Every model predicts the expected case count for an area in the target month. A
negative-binomial predictive distribution around that mean gives the intervals and the
probability of a surge, so all models are compared on the same footing.

Implemented in this prototype: negative binomial GLM, quasi-Poisson GLM, Random Forest,
XGBoost, and a seasonal-naive baseline. LSTM is planned for when the real 5+ year
facility series are available (it needs more data than the prototype simulates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from .features import FEATURES

# Features the GLMs use. Lagged rates enter on the log scale.
LOG_FEATURES = ["rate_lag0", "rate_lag1", "rate_lag2", "rate_same_month_ly", "opd_rate"]


def _moment_alpha(y, mu) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    a = np.mean(((y - mu) ** 2 - mu) / np.maximum(mu, 1e-6) ** 2)
    return float(np.clip(a, 0.005, 5))


class BaseModel:
    name = "base"
    alpha: float = 0.1

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # --- predictive distribution -------------------------------------------------
    def _nb_params(self, mu):
        mu = np.maximum(np.asarray(mu, float), 1e-6)
        n = 1.0 / self.alpha
        return n, n / (n + mu)

    def interval(self, mu, level: float):
        n, p = self._nb_params(mu)
        lo = stats.nbinom.ppf((1 - level) / 2, n, p)
        hi = stats.nbinom.ppf(1 - (1 - level) / 2, n, p)
        return lo, hi

    def prob_exceed(self, mu, threshold):
        n, p = self._nb_params(mu)
        return stats.nbinom.sf(np.asarray(threshold, float), n, p)

    def sample(self, mu, size: int, rng):
        n, p = self._nb_params(mu)
        return rng.negative_binomial(n, p, size=(size, len(np.atleast_1d(mu))))

    def _calibrate_alpha(self, df: pd.DataFrame):
        """Estimate dispersion on the latest 20% of training months, then refit on all."""
        cut = df["target_month"].quantile(0.8)
        early, late = df[df.target_month <= cut], df[df.target_month > cut]
        if len(late) > 30 and len(early) > 60:
            self._fit_mean(early)
            self.alpha = _moment_alpha(late["y"], self.predict_mean(late))
        self._fit_mean(df)
        return self


class NegBinGLM(BaseModel):
    name = "Negative binomial (GLM)"
    family_name = "nb"

    def _design(self, df):
        X = df[FEATURES].copy()
        for c in LOG_FEATURES:
            X[c] = np.log(X[c] + 0.1)
        return X

    def fit(self, df):
        X = self._design(df)
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1)
        Xs = sm.add_constant((X - self.mu_) / self.sd_, has_constant="add")
        off = np.log(df["pop_target"] / 1000)
        pois = sm.GLM(df["y"], Xs, family=sm.families.Poisson(), offset=off).fit()
        if self.family_name == "nb":
            self.alpha = _moment_alpha(df["y"], pois.mu)
            self.res_ = sm.GLM(df["y"], Xs, family=sm.families.NegativeBinomial(alpha=self.alpha),
                               offset=off).fit()
        else:
            self.res_ = sm.GLM(df["y"], Xs, family=sm.families.Poisson(), offset=off).fit(scale="X2")
            self.alpha = _moment_alpha(df["y"], self.res_.mu)
        return self

    def predict_mean(self, df):
        X = self._design(df)
        Xs = sm.add_constant((X - self.mu_) / self.sd_, has_constant="add")
        return np.asarray(self.res_.predict(Xs, offset=np.log(df["pop_target"] / 1000)))

    def coefficients(self) -> pd.DataFrame:
        ci = self.res_.conf_int()
        out = pd.DataFrame({"feature": self.res_.params.index, "coef": self.res_.params.values,
                            "lo": ci[0].values, "hi": ci[1].values, "p": self.res_.pvalues.values})
        out = out[out.feature != "const"]
        out[["irr", "irr_lo", "irr_hi"]] = np.exp(out[["coef", "lo", "hi"]])
        return out


class QuasiPoissonGLM(NegBinGLM):
    name = "Quasi-Poisson (GLM)"
    family_name = "qp"


class RandomForestModel(BaseModel):
    name = "Random Forest"

    def _fit_mean(self, df):
        self.m_ = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                        n_jobs=-1, random_state=0)
        self.m_.fit(df[FEATURES], 1000 * df["y"] / df["pop_target"])

    def fit(self, df):
        return self._calibrate_alpha(df)

    def predict_mean(self, df):
        return np.maximum(self.m_.predict(df[FEATURES]), 0.01) * df["pop_target"].values / 1000

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": FEATURES, "importance": self.m_.feature_importances_})


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def _fit_mean(self, df):
        self.m_ = xgb.XGBRegressor(objective="count:poisson", n_estimators=250, max_depth=4,
                                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=3, n_jobs=-1, random_state=0)
        self.m_.fit(df[FEATURES], df["y"], base_margin=np.log(df["pop_target"] / 1000))

    def fit(self, df):
        return self._calibrate_alpha(df)

    def predict_mean(self, df):
        d = xgb.DMatrix(df[FEATURES], base_margin=np.log(df["pop_target"] / 1000))
        return self.m_.get_booster().predict(d)

    def shap_values(self, df) -> pd.DataFrame:
        """Exact TreeSHAP contributions on the log scale (last column = bias)."""
        d = xgb.DMatrix(df[FEATURES], base_margin=np.log(df["pop_target"] / 1000))
        contrib = self.m_.get_booster().predict(d, pred_contribs=True)
        return pd.DataFrame(contrib, columns=FEATURES + ["bias"], index=df.index)


class SeasonalNaive(BaseModel):
    name = "Seasonal naive (baseline)"

    def fit(self, df):
        self.alpha = _moment_alpha(df["y"], self.predict_mean(df))
        return self

    def predict_mean(self, df):
        return np.maximum(df["rate_same_month_ly"].values * df["pop_target"].values / 1000, 0.5)


MODELS = {m.name: m for m in [NegBinGLM, XGBoostModel, QuasiPoissonGLM, RandomForestModel, SeasonalNaive]}
# NB GLM is the default: fast to train and coefficient-based, so easiest to explain.
DEFAULT_MODEL = NegBinGLM.name


def make_model(name: str) -> BaseModel:
    return MODELS[name]()
