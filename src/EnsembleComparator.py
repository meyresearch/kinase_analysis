from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import ast

from collections.abc import Iterable

import joblib
import mdtraj as md
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    featurize_trajectories,
    feature_cleanup_with_stats,
    get_feature_atoms,
)


@dataclass
class EnsembleComparatorConfig:
    """Configuration for ensemble comparison workflows."""

    work_dir: Path
    feature_cache_dir: Path
    model_dir: Path
    plot_dir: Path
    report_dir: Path
    metadata_path: Path


class EnsembleComparator:
    """Compare two structural ensembles via feature-based ML models.

    This class consolidates the notebook workflow into reusable steps:

    1. Load mdtraj trajectories and featurize them.
    2. Perform feature selection and persist selected features.
    3. Train logistic regression and random forest classifiers.
    4. Generate feature-importance visualisations.
    5. Reload selected features, labels, and trained models from disk.
    """

    DEFAULT_LOGISTIC_PARAM_GRID: Dict[str, Sequence] = {
        "anova__k": [300, 500, 800, 1000],
        "sparse__estimator__C": [0.05, 0.1, 0.2, 0.5, 1.0],
        "clf__C": [0.5, 1.0, 2.0],
    }

    DEFAULT_RF_PARAM_GRID: Dict[str, Sequence] = {
        "anova__k": [300, 500, 800, 1000],
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [10, 15, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    def __init__(
        self,
        work_dir: Optional[Path | str] = None,
    ) -> None:
        base = Path(work_dir) if work_dir else Path.cwd() / "ensemble_comparator"

        self.config = EnsembleComparatorConfig(
            work_dir=base,
            feature_cache_dir=base / "features",
            model_dir=base / "models",
            plot_dir=base / "plots",
            report_dir=base / "reports",
            metadata_path=base / "features" / "metadata.json",
        )

        for directory in (
            self.config.feature_cache_dir,
            self.config.model_dir,
            self.config.plot_dir,
            self.config.report_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.s1_features: Optional[np.ndarray] = None
        self.s2_features: Optional[np.ndarray] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.X_selected: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_info: Optional[Dict[str, Tuple[int, int]]] = None
        self.atom_indices_info: Optional[Dict[str, List[Tuple[int, ...]]]] = None
        self.reference_traj: Optional[md.Trajectory] = None
        self.metadata: Dict = {}

        self.logistic_model: Optional[Pipeline] = None
        self.random_forest_model: Optional[Pipeline] = None
        self.logistic_feature_importance: Optional[pd.DataFrame] = None
        self.logistic_permutation_importance: Optional[pd.DataFrame] = None
        self.random_forest_feature_importance: Optional[pd.DataFrame] = None
        self.random_forest_permutation_importance: Optional[pd.DataFrame] = None


    def load_and_featurize(
        self,
        ensemble1_sources: Sequence[Path | str],
        ensemble2_sources: Sequence[Path | str],
        pattern: str = "*.pdb",
        force: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load trajectories for two ensembles and featurize them.

        Parameters
        ----------
        ensemble1_sources, ensemble2_sources
            Iterable of files or directories containing trajectory files.
        pattern
            Glob used when an entry is a directory.
        force
            If ``True`` recompute features even when cached arrays exist.
        """

        if not force and self._features_cached():
            self._load_cached_features()
            print(f"Loaded cached features from {self.config.feature_cache_dir}")
            return self.s1_features, self.s2_features

        files_1 = self._collect_traj_files(ensemble1_sources, pattern)
        files_2 = self._collect_traj_files(ensemble2_sources, pattern)

        if not files_1 or not files_2:
            raise ValueError("Both ensembles must provide at least one trajectory file.")

        s1_traj = [md.load(str(path)) for path in files_1]
        s2_traj = [md.load(str(path)) for path in files_2]

        self.reference_traj = s1_traj[0]

        s1_features, feature_info, atom_indices_info = featurize_trajectories(s1_traj)
        s2_features, _, _ = featurize_trajectories(s2_traj)
        print(f"Featurized {len(files_1)} files {len(s1_features)} frames for ensemble 1, "
              f"{len(files_2)} files {len(s2_features)} frames for ensemble 2.")

        self.s1_features = s1_features
        self.s2_features = s2_features
        self.feature_info = feature_info
        self.atom_indices_info = atom_indices_info

        self.metadata = {
            "ensemble1_files": [str(p) for p in files_1],
            "ensemble2_files": [str(p) for p in files_2],
            "reference_traj": str(files_1[0]),
        }

        self._cache_features()
        return self.s1_features, self.s2_features


    def select_features(
        self,
        var_threshold: float = 1e-3,
        corr_threshold: float = 0.85,
        force: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply variance/correlation cleanup and persist the result."""

        if self.s1_features is None or self.s2_features is None:
            print("Features not found; loading cached features first.")
            self._load_cached_features(require_reference=True)

        selected_path = self.config.feature_cache_dir / "selected_idx.npy"
        if not force and selected_path.exists():
            idx = np.load(selected_path)
            X, y = self._build_selected_dataset(idx)
            self.selected_idx = idx
            self.X_selected = X
            self.y = y
            print(f"Loaded selected features from {selected_path}")
            return idx, X, y

        idx = feature_cleanup_with_stats(
            self.s1_features,
            self.s2_features,
            var_threshold=var_threshold,
            corr_threshold=corr_threshold,
        )
        np.save(selected_path, idx)
        print(f"Selected {len(idx)} features after cleanup.")

        X, y = self._build_selected_dataset(idx)
        self.selected_idx = idx
        self.X_selected = X
        self.y = y
        return idx, X, y


    def train_classifiers(
        self,
        train_logistic: bool = True,
        train_random_forest: bool = True,
        logistic_param_grid: Optional[Dict[str, Sequence]] = None,
        rf_param_grid: Optional[Dict[str, Sequence]] = None,
        logistic_cv_splits: int = 5,
        rf_cv_splits: int = 5,
        scoring: str = "roc_auc",
        force: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Train classifiers on the selected feature set and persist artefacts."""

        if self.X_selected is None or self.y is None:
            print("Selected features not found; running feature selection first.")
            self.select_features()
            
        results: Dict[str, Dict[str, float]] = {}

        if train_logistic:
            results["logistic"] = self._train_logistic(
                param_grid=logistic_param_grid or self.DEFAULT_LOGISTIC_PARAM_GRID,
                cv_splits=logistic_cv_splits,
                scoring=scoring,
                force=force,
            )
            print("Trained logistic regression model.")

        if train_random_forest:
            results["random_forest"] = self._train_random_forest(
                param_grid=rf_param_grid or self.DEFAULT_RF_PARAM_GRID,
                cv_splits=rf_cv_splits,
                scoring=scoring,
                force=force,
            )
            print("Trained random forest model.")
        return results


    def plot_feature_importance(
        self,
        model_type: str = "logistic",
        top_k: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Generate feature-importance plots for the requested model.

        Parameters
        ----------
        model_type : str
            "logistic" or "random_forest".
        top_k : Optional[int]
            Optionally limit scatter plots to the top-k features by absolute importance.
        """

        model_type = model_type.lower()
        if model_type not in {"logistic", "random_forest"}:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")

        if self.reference_traj is None:
            self._ensure_reference_traj_loaded()

        if model_type == "logistic":
            if self.logistic_model is None:
                self._load_logistic_from_disk()
            if self.logistic_feature_importance is None:
                self._load_logistic_importance()
            if self.logistic_feature_importance is None:
                raise RuntimeError("Logistic feature importance data is unavailable.")
            outputs: Dict[str, Path] = {}
            outputs.update(
                self._render_feature_importance_plots(
                    feature_df=self.logistic_feature_importance,
                    value_col="coefficient",
                    title_prefix="Logistic Regression",
                    heatmap_cmap="bwr",
                    base_filename="logistic",
                    cbar_label="Logistic Regression Coefficient",
                    symmetric=True,
                    top_k=top_k,
                )
            )

            if self.logistic_permutation_importance is None:
                self._load_logistic_permutation_importance()
            if self.logistic_permutation_importance is None:
                self.compute_logistic_permutation_importance()

            if self.logistic_permutation_importance is not None:
                outputs.update(
                    self._render_feature_importance_plots(
                        feature_df=self.logistic_permutation_importance,
                        value_col="permutation_importance",
                        title_prefix="Logistic Regression (Permutation)",
                        heatmap_cmap="OrRd",
                        base_filename="logistic_permutation",
                        cbar_label="Logistic Permutation Importance",
                        symmetric=False,
                        top_k=top_k,
                    )
                )

            return outputs

        if self.random_forest_model is None:
            self._load_random_forest_from_disk()
        if self.random_forest_feature_importance is None:
            self._load_random_importance()
        if self.random_forest_feature_importance is None:
            raise RuntimeError("Random forest feature importance data is unavailable.")

        outputs: Dict[str, Path] = {}
        outputs.update(
            self._render_feature_importance_plots(
                feature_df=self.random_forest_feature_importance,
                value_col="importance",
                title_prefix="Random Forest",
                heatmap_cmap="Reds",
                base_filename="random_forest_model",
                cbar_label="Random Forest Importance",
                symmetric=False,
                top_k=top_k,
            )
        )

        if self.random_forest_permutation_importance is None:
            self._load_random_permutation_importance()
        if self.random_forest_permutation_importance is None:
            self.compute_random_forest_permutation_importance()

        outputs.update(
            self._render_feature_importance_plots(
                feature_df=self.random_forest_permutation_importance,
                value_col="permutation_importance",
                title_prefix="Random Forest (Permutation)",
                heatmap_cmap="Reds",
                base_filename="random_forest_permutation",
                cbar_label="Random Forest Permutation Importance",
                symmetric=False,
                top_k=top_k,
            )
        )
        return outputs
    

    def recover_from_disk(self) -> Dict[str, object]:
        """Load cached arrays and trained models from the working directory."""

        artefacts: Dict[str, object] = {}

        if self._features_cached():
            self._load_cached_features()
            artefacts["s1_features"] = self.s1_features
            artefacts["s2_features"] = self.s2_features
            artefacts["feature_info"] = self.feature_info
            artefacts["atom_indices_info"] = self.atom_indices_info

        selected_path = self.config.feature_cache_dir / "selected_idx.npy"

        if selected_path.exists():
            self.selected_idx = np.load(selected_path)
            artefacts["selected_idx"] = self.selected_idx
            if self.s1_features is not None and self.s2_features is not None:
                X, y = self._build_selected_dataset(self.selected_idx)
                self.X_selected = X
                self.y = y
                artefacts["X"] = X
                artefacts["y"] = y

        logistic_path = self.config.model_dir / "logistic_pipeline.joblib"
        if logistic_path.exists():
            self.logistic_model = joblib.load(logistic_path)
            artefacts["logistic_model"] = self.logistic_model
            self._load_logistic_importance()
            artefacts["logistic_feature_importance"] = self.logistic_feature_importance
            self._load_logistic_permutation_importance()
            artefacts["logistic_permutation_importance"] = self.logistic_permutation_importance

        rf_path = self.config.model_dir / "random_forest_pipeline.joblib"
        if rf_path.exists():
            self.random_forest_model = joblib.load(rf_path)
            artefacts["random_forest_model"] = self.random_forest_model
            self._load_random_importance()
            artefacts["random_forest_feature_importance"] = (
                self.random_forest_feature_importance
            )
            self._load_random_permutation_importance()
            artefacts["random_forest_permutation_importance"] = (
                self.random_forest_permutation_importance
            )

        return artefacts


    @classmethod
    def load(
        cls,
        work_dir: Optional[Path | str] = None,
        auto_recover: bool = True,
    ) -> "EnsembleComparator":
        """Instantiate a comparator and optionally recover cached artefacts."""

        instance = cls(work_dir=work_dir)
        if auto_recover:
            instance.recover_from_disk()
        return instance


    def _train_logistic(
        self,
        param_grid: Dict[str, Sequence],
        cv_splits: int,
        scoring: str,
        force: bool,
    ) -> Dict[str, float]:
        model_path = self.config.model_dir / "logistic_pipeline.joblib"
        summary_path = self.config.report_dir / "logistic_summary.json"

        if not force and model_path.exists():
            self.logistic_model = joblib.load(model_path)
            self._load_logistic_importance()
            with open(summary_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("anova", SelectKBest(score_func=f_classif, k=1000)),
            ("sparse", SelectFromModel(
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=4000,
                    n_jobs=-1,
                ),
                max_features=None,
                threshold="median",
            )),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000)),
        ])

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid.fit(self.X_selected, self.y)

        self.logistic_model = grid.best_estimator_
        joblib.dump(self.logistic_model, model_path)

        summary = {
            "best_score": float(grid.best_score_),
            "best_params": grid.best_params_,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        self._save_logistic_importance()
        return summary


    def _train_random_forest(
        self,
        param_grid: Dict[str, Sequence],
        cv_splits: int,
        scoring: str,
        force: bool,
    ) -> Dict[str, float]:
        model_path = self.config.model_dir / "random_forest_pipeline.joblib"
        summary_path = self.config.report_dir / "random_forest_summary.json"

        if not force and model_path.exists():
            self.random_forest_model = joblib.load(model_path)
            self._load_random_importance()
            self._load_random_permutation_importance()
            with open(summary_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("anova", SelectKBest(score_func=f_classif, k=1000)),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ])

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid.fit(self.X_selected, self.y)

        self.random_forest_model = grid.best_estimator_
        joblib.dump(self.random_forest_model, model_path)

        summary = {
            "best_score": float(grid.best_score_),
            "best_params": grid.best_params_,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        self._save_random_importance()
        return summary


    def _collect_traj_files(
        self,
        sources: Sequence[Path | str],
        pattern: str,
    ) -> List[Path]:
        files: List[Path] = []
        for src in sources:
            path = Path(src)
            if path.is_dir():
                files.extend(sorted(path.glob(pattern)))
            elif path.is_file():
                files.append(path)
            else:
                raise FileNotFoundError(f"Trajectory source not found: {src}")
        return files


    def _features_cached(self) -> bool:
        base = self.config.feature_cache_dir
        required = [
            base / "s1_features.npy",
            base / "s2_features.npy",
            base / "feature_info.pkl",
            base / "atom_indices_info.pkl",
        ]
        return all(path.exists() for path in required)


    def _cache_features(self) -> None:
        base = self.config.feature_cache_dir
        np.save(base / "s1_features.npy", self.s1_features)
        np.save(base / "s2_features.npy", self.s2_features)

        with open(base / "feature_info.pkl", "wb") as f_info:
            pickle.dump(self.feature_info, f_info)
        with open(base / "atom_indices_info.pkl", "wb") as f_atoms:
            pickle.dump(self.atom_indices_info, f_atoms)

        with open(self.config.metadata_path, "w", encoding="utf-8") as f_meta:
            json.dump(self.metadata, f_meta, indent=2)

        print(f"Cached features to {base}")


    def _load_cached_features(self, require_reference: bool = False) -> None:
        base = self.config.feature_cache_dir
        self.s1_features = np.load(base / "s1_features.npy")
        self.s2_features = np.load(base / "s2_features.npy")
        with open(base / "feature_info.pkl", "rb") as f_info:
            self.feature_info = pickle.load(f_info)
        with open(base / "atom_indices_info.pkl", "rb") as f_atoms:
            self.atom_indices_info = pickle.load(f_atoms)

        if self.config.metadata_path.exists():
            with open(self.config.metadata_path, "r", encoding="utf-8") as f_meta:
                self.metadata = json.load(f_meta)
        if require_reference:
            self._ensure_reference_traj_loaded()

    def _ensure_reference_traj_loaded(self) -> None:
        if self.reference_traj is not None:
            return
        if not self.metadata or "reference_traj" not in self.metadata:
            raise RuntimeError("Reference trajectory path not found in metadata.")
        ref_path = Path(self.metadata["reference_traj"]).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference trajectory file missing: {self.metadata['reference_traj']}"
            )
        self.reference_traj = md.load(str(ref_path))

    def _build_selected_dataset(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.s1_features is None or self.s2_features is None:
            self._load_cached_features(require_reference=False)

        s1 = self.s1_features
        s2 = self.s2_features
        if s1 is None or s2 is None:
            raise RuntimeError("Feature arrays are unavailable; load features before building dataset.")

        idx = np.asarray(idx, dtype=int)
        X = np.concatenate([s1, s2], axis=0)[:, idx]
        y = np.array([0] * len(s1) + [1] * len(s2))
        return X, y

    def _build_feature_dataframe(
        self,
        feature_importance: List[Tuple[int, float]],
        value_name: str,
    ) -> pd.DataFrame:
        rows = []
        for feature_idx, value in feature_importance:
            feature_type, description, atom_indices = get_feature_atoms(
                feature_idx,
                self.feature_info,
                self.atom_indices_info,
                self.reference_traj,
            )
            rows.append(
                {
                    "feature_idx": feature_idx,
                    value_name: value,
                    "abs_value": abs(value),
                    "feature_type": feature_type,
                    "description": description,
                    "atom_indices": list(atom_indices)
                    if isinstance(atom_indices, Iterable)
                    else atom_indices,
                }
            )
        return pd.DataFrame(rows)

    def _save_logistic_importance(self) -> None:
        anova = self.logistic_model.named_steps["anova"]
        sparse = self.logistic_model.named_steps["sparse"]
        clf = self.logistic_model.named_steps["clf"]

        mask_anova = anova.get_support()
        selected_idx_anova = self.selected_idx[mask_anova]
        mask_sparse = sparse.get_support()
        final_idx = selected_idx_anova[mask_sparse]

        coef = clf.coef_[0]
        importance = list(zip(final_idx, coef))

        if self.reference_traj is None:
            self._ensure_reference_traj_loaded()

        df = self._build_feature_dataframe(importance, "coefficient")
        df.sort_values("abs_value", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(self.config.report_dir / "logistic_feature_importance.csv", index=False)
        self.logistic_feature_importance = df
        self.compute_logistic_permutation_importance(force=True)


    def _save_random_importance(self) -> None:
        anova = self.random_forest_model.named_steps["anova"]
        rf = self.random_forest_model.named_steps["rf"]

        mask_anova = anova.get_support()
        selected_idx_anova = self.selected_idx[mask_anova]
        importance = list(zip(selected_idx_anova, rf.feature_importances_))

        if self.reference_traj is None:
            self._ensure_reference_traj_loaded()

        df = self._build_feature_dataframe(importance, "importance")
        df.sort_values("importance", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(self.config.report_dir / "random_forest_feature_importance.csv", index=False)
        self.random_forest_feature_importance = df
        self.compute_random_forest_permutation_importance(force=True)


    def compute_logistic_permutation_importance(
        self,
        n_repeats: int = 10,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        force: bool = False,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation-based feature importance for the logistic classifier."""

        if self.logistic_model is None:
            self._load_logistic_from_disk()

        if self.X_selected is None or self.y is None:
            self.select_features()

        path = self.config.report_dir / "logistic_permutation_importance.csv"
        if not force and path.exists():
            df = pd.read_csv(path)
            self.logistic_permutation_importance = df
            return df

        result = permutation_importance(
            self.logistic_model,
            self.X_selected,
            self.y,
            scoring=scoring,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        anova = self.logistic_model.named_steps["anova"]
        sparse = self.logistic_model.named_steps["sparse"]

        mask_anova = anova.get_support()
        selected_idx_anova = self.selected_idx[mask_anova]
        mean_importance = result.importances_mean[mask_anova]
        std_importance = result.importances_std[mask_anova]

        mask_sparse = sparse.get_support()
        final_idx = selected_idx_anova[mask_sparse]
        mean_importance = mean_importance[mask_sparse]
        std_importance = std_importance[mask_sparse]

        data = list(zip(final_idx, mean_importance))

        if self.reference_traj is None:
            self._ensure_reference_traj_loaded()

        df = self._build_feature_dataframe(data, "permutation_importance")
        df["importance_std"] = std_importance
        df.sort_values("permutation_importance", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path, index=False)
        self.logistic_permutation_importance = df
        return df


    def compute_random_forest_permutation_importance(
        self,
        n_repeats: int = 10,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        force: bool = False,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation-based feature importance for the random forest classifier."""

        if self.random_forest_model is None:
            self._load_random_forest_from_disk()

        if self.X_selected is None or self.y is None:
            self.select_features()

        path = self.config.report_dir / "random_forest_permutation_importance.csv"
        if not force and path.exists():
            df = pd.read_csv(path)
            self.random_forest_permutation_importance = df
            return df

        result = permutation_importance(
            self.random_forest_model,
            self.X_selected,
            self.y,
            scoring=scoring,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        anova = self.random_forest_model.named_steps["anova"]
        mask_anova = anova.get_support()
        selected_idx_anova = self.selected_idx[mask_anova]

        mean_importance = result.importances_mean[mask_anova]
        std_importance = result.importances_std[mask_anova]

        data = list(zip(selected_idx_anova, mean_importance))

        if self.reference_traj is None:
            self._ensure_reference_traj_loaded()

        df = self._build_feature_dataframe(data, "permutation_importance")
        df["importance_std"] = std_importance
        df.sort_values("permutation_importance", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path, index=False)
        self.random_forest_permutation_importance = df
        return df


    def _load_logistic_from_disk(self) -> None:
        model_path = self.config.model_dir / "logistic_pipeline.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Logistic regression model has not been trained yet.")
        self.logistic_model = joblib.load(model_path)
        self._load_logistic_importance()
        self._load_logistic_permutation_importance()


    def _load_random_forest_from_disk(self) -> None:
        model_path = self.config.model_dir / "random_forest_pipeline.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Random forest model has not been trained yet.")
        self.random_forest_model = joblib.load(model_path)
        self._load_random_importance()
        self._load_random_permutation_importance()


    def _load_logistic_importance(self) -> None:
        path = self.config.report_dir / "logistic_feature_importance.csv"
        if path.exists():
            self.logistic_feature_importance = pd.read_csv(path)


    def _load_logistic_permutation_importance(self) -> None:
        path = self.config.report_dir / "logistic_permutation_importance.csv"
        if path.exists():
            self.logistic_permutation_importance = pd.read_csv(path)


    def _load_random_importance(self) -> None:
        path = self.config.report_dir / "random_forest_feature_importance.csv"
        if path.exists():
            self.random_forest_feature_importance = pd.read_csv(path)


    def _load_random_permutation_importance(self) -> None:
        path = self.config.report_dir / "random_forest_permutation_importance.csv"
        if path.exists():
            self.random_forest_permutation_importance = pd.read_csv(path)


    def _build_ca_matrix(
        self,
        feature_df: pd.DataFrame,
        value_col: str,
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        ca_df = feature_df[feature_df["feature_type"] == "ca_distances"]
        if ca_df.empty:
            return None, None

        residues = list(self.reference_traj.topology.residues)
        first_res_id = residues[0].resSeq
        n_res = len(residues)
        matrix = np.zeros((n_res, n_res))
        residue_labels = [res.resSeq for res in residues]

        for _, row in ca_df.iterrows():
            feature_idx = int(row["feature_idx"])
            value_raw = row[value_col]
            if pd.isna(value_raw):
                continue
            value = float(value_raw)
            _, _, atom_indices = get_feature_atoms(
                feature_idx,
                self.feature_info,
                self.atom_indices_info,
                self.reference_traj,
            )
            atom0, atom1 = atom_indices
            res0 = self.reference_traj.topology.atom(atom0).residue.resSeq - first_res_id
            res1 = self.reference_traj.topology.atom(atom1).residue.resSeq - first_res_id
            matrix[res0, res1] = value
            matrix[res1, res0] = value

        return matrix, residue_labels


    def _plot_dihedral_scatter(
        self,
        feature_df: pd.DataFrame,
        value_col: str,
        title_prefix: str,
        top_k: Optional[int],
        value_label: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        dihedral_mask = feature_df["feature_type"].str.contains("phi|psi|chi", case=False, regex=True)
        dihedral_df = feature_df[dihedral_mask]
        if dihedral_df.empty:
            return None

        if top_k is not None and top_k < len(dihedral_df):
            dihedral_df = dihedral_df.nlargest(top_k, columns="abs_value")

        def _residue_from_indices(indices: Sequence[int]) -> int:
            if not isinstance(indices, list):
                indices = ast.literal_eval(indices)
            atom = self.reference_traj.topology.atom(indices[1 if len(indices) > 1 else 0])
            return atom.residue.resSeq

        dihedral_df = dihedral_df.copy()
        dihedral_df["residue"] = dihedral_df["atom_indices"].apply(_residue_from_indices)

        phi_df = dihedral_df[dihedral_df["feature_type"].str.contains("phi")]
        psi_df = dihedral_df[dihedral_df["feature_type"].str.contains("psi")]
        chi_df = dihedral_df[dihedral_df["feature_type"].str.contains("chi")]

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        plots = [(phi_df, axes[0], "Phi (φ)"), (psi_df, axes[1], "Psi (ψ)"), (chi_df, axes[2], "Chi (χ)")]

        y_label = value_label or value_col.replace("_", " ").title()

        for df, ax, label in plots:
            if df.empty:
                ax.set_visible(False)
                continue
            colors = {
                "sin": "tab:red",
                "cos": "tab:blue",
                "chi1": "tab:green",
                "chi2": "tab:orange",
                "chi3": "tab:purple",
                "chi4": "tab:brown",
            }
            for feature_type, subset in df.groupby("feature_type"):
                color_key = "sin" if feature_type.endswith("sin") else "cos"
                if "chi" in feature_type:
                    color_key = feature_type.split("_")[0]
                ax.scatter(
                    subset["residue"],
                    subset[value_col],
                    s=50,
                    alpha=0.7,
                    label=feature_type,
                    color=colors.get(color_key, "tab:gray"),
                    marker="o" if feature_type.endswith("sin") else "s",
                )
            ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
            ax.set_ylabel(y_label)
            ax.set_title(f"{label} Dihedral Angles")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        axes[-1].set_xlabel("Residue Number")
        # plt.suptitle(f"{title_prefix}: Dihedral Features", y=0.92)
        plt.tight_layout()
        return fig


    def _render_feature_importance_plots(
        self,
        feature_df: pd.DataFrame,
        value_col: str,
        title_prefix: str,
        heatmap_cmap: str,
        base_filename: str,
        cbar_label: str,
        symmetric: bool,
        top_k: Optional[int],
    ) -> Dict[str, Path]:
        """Create heatmap and dihedral scatter plots for a feature-importance table."""

        output_paths: Dict[str, Path] = {}
        matrix, residue_labels = self._build_ca_matrix(feature_df, value_col)
        if matrix is not None:
            fig, ax = plt.subplots(figsize=(12, 10))
            if symmetric:
                vmax = np.max(np.abs(matrix))
                vmin = -vmax
            else:
                vmax = matrix.max()
                vmin = 0.0
                if vmax <= 0:
                    vmax = 1e-12
            im = ax.imshow(matrix, cmap=heatmap_cmap, aspect="auto", vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(cbar_label, rotation=270, labelpad=20)
            ax.set_xlabel("Residue Index")
            ax.set_ylabel("Residue Index")
            ax.set_title(f"{title_prefix}: CA Distance Features")
            ticks = np.arange(0, len(residue_labels), max(1, len(residue_labels) // 20))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([residue_labels[i] for i in ticks])
            ax.set_yticklabels([residue_labels[i] for i in ticks])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            heatmap_path = self.config.plot_dir / f"{base_filename}_ca_heatmap.png"
            fig.savefig(heatmap_path, dpi=300)
            plt.close(fig)
            output_paths[f"{base_filename}_ca_heatmap"] = heatmap_path

        dihedral_fig = self._plot_dihedral_scatter(
            feature_df=feature_df,
            value_col=value_col,
            title_prefix=title_prefix,
            top_k=top_k,
            value_label=cbar_label,
        )
        if dihedral_fig is not None:
            scatter_path = self.config.plot_dir / f"{base_filename}_dihedral_scatter.png"
            dihedral_fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(dihedral_fig)
            output_paths[f"{base_filename}_dihedral_scatter"] = scatter_path

        return output_paths

