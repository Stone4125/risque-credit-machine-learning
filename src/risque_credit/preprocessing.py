"""Imputation ajustee uniquement sur les donnees d'entrainement."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ImputationParGroupe(BaseEstimator, TransformerMixin):
    """Impute le taux d'interet par grade et l'anciennete par la mediane du train.

    Les statistiques sont calculees dans ``fit``. Un pli de validation ou un
    dossier nouveau ne contribuent pas a ces medianes.
    """

    def fit(self, X, y=None):
        cadre = self._cadre(X)
        self.n_features_in_ = cadre.shape[1]
        self.feature_names_in_ = list(cadre.columns)
        self.mediane_taux_par_grade_ = cadre.groupby("loan_grade")["loan_int_rate"].median()
        self.mediane_taux_ = float(cadre["loan_int_rate"].median())
        self.mediane_anciennete_ = float(cadre["person_emp_length"].median())
        return self

    def transform(self, X):
        cadre = self._cadre(X).copy()
        taux_par_grade = cadre["loan_grade"].map(self.mediane_taux_par_grade_)
        cadre["loan_int_rate"] = (
            cadre["loan_int_rate"].fillna(taux_par_grade).fillna(self.mediane_taux_)
        )
        cadre["person_emp_length"] = cadre["person_emp_length"].fillna(self.mediane_anciennete_)
        return cadre

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)

    def _cadre(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if not hasattr(self, "feature_names_in_"):
            raise TypeError("ImputationParGroupe attend un DataFrame a l'ajustement.")
        return pd.DataFrame(X, columns=self.feature_names_in_)
