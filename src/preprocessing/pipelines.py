"""Recettes de l'étape 4 pour comparer le traitement commun et les stratégies de réduction des redondances.

Chaque recette est un Pipeline scikit-learn de prétraitement, sans modèle : on lui
ajoute un estimateur avec make_pipeline(make_base(), RidgeCV()). Tout ce qui est
appris (médianes, colonnes passées au log, moyennes, catégories, colonnes
sélectionnées, axes de l'ACP) l'est dans fit, donc sur l'entraînement de la
partition ou du pli seulement.

Le retrait des deux ventes aberrantes (remove_outliers) n'est pas une étape du
pipeline : un transformateur scikit-learn ne peut pas retirer des lignes de y.
Il s'applique aux données d'entraînement avant fit, jamais à la validation ni à test.

Les cinq recettes produisent un DataFrame dont les colonnes ont un nom.
"""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline

from src.preprocessing.steps.cleaning import Cleaner
from src.preprocessing.steps.encoding import QualityOrdinalEncoder, make_onehot_encoder, nominal_columns
from src.preprocessing.steps.features import FeatureCreator
from src.preprocessing.steps.missing import MissingValueHandler
from src.preprocessing.steps.scaling import SkewCorrector, make_scaler
from src.preprocessing.strategies.causal import CAUSAL_MODE, CausalSelector
from src.preprocessing.strategies.correlation import (
    MAX_PAIR_CORRELATION, MIN_TARGET_CORRELATION, CorrelationSelector,
)
from src.preprocessing.strategies.pca import BLOCKS, PCA_VARIANCE, BlockPCA, PCAReducer


def _prepare(*selection):
    """Nettoyage, valeurs manquantes, création, encodage ordinal, puis sélection éventuelle."""
    return [Cleaner(), MissingValueHandler(), FeatureCreator(), QualityOrdinalEncoder(), *selection]


def _encode_and_scale(*numeric_reduction, log=True, scaler="standard"):
    """log1p des longues queues, standardisation des colonnes numériques et one-hot des nominales."""
    columns = ColumnTransformer(
        [("numeric", make_pipeline(make_scaler(scaler), *numeric_reduction),
          make_column_selector(dtype_include="number")),
         ("nominal", make_onehot_encoder(), nominal_columns)],
        verbose_feature_names_out=False,
    )
    return [*([SkewCorrector()] if log else []), columns.set_output(transform="pandas")]


def make_base(log=True, scaler="standard"):
    """Construire le pipeline avec les cinq étapes communes."""
    return make_pipeline(*_prepare(), *_encode_and_scale(log=log, scaler=scaler))


def make_correlation(min_target_corr=MIN_TARGET_CORRELATION, max_pair_corr=MAX_PAIR_CORRELATION):
    """Construire le pipeline avec sélection par corrélation.

    La sélection vient après l'encodage : elle juge chaque colonne one-hot séparément,
    et chaque échelle ordinale en entier.
    """
    return make_pipeline(*_prepare(), *_encode_and_scale(),
                         CorrelationSelector(min_target_corr, max_pair_corr))


def make_causal(mode=CAUSAL_MODE):
    """Construire le pipeline avec sélection causale."""
    return make_pipeline(*_prepare(CausalSelector(mode)), *_encode_and_scale())


def make_pca(n_components=PCA_VARIANCE):
    """Construire le pipeline avec analyse en composantes principales.

    L'ACP porte sur les colonnes numériques standardisées. Les colonnes one-hot,
    presque vides, ne s'y prêtent pas et restent telles quelles.
    """
    return make_pipeline(*_prepare(), *_encode_and_scale(PCAReducer(n_components)))


def make_causal_pca(mode=CAUSAL_MODE, blocks=BLOCKS):
    """Construire le pipeline combinant sélection causale et ACP par blocs."""
    return make_pipeline(*_prepare(CausalSelector(mode)), *_encode_and_scale(), BlockPCA(blocks))


RECIPES = {
    "Base": make_base,
    "Corrélation": make_correlation,
    "Causale": make_causal,
    "ACP": make_pca,
    "Causale + ACP par blocs": make_causal_pca,
}
