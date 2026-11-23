"""Shared helpers for Session 7 pool-based BO demos (BH and START).

Run scripts or notebooks with cwd = BH, START, or this session folder so imports resolve.
Lesson tuning: ``BH_POOL_BO_*`` / ``bh_pool_bo_*`` for BH.py; ``START_POOL_BO_*`` / ``start_pool_bo_*`` for pool_bo_beginner.py.

Pool BO pieces (compose a step loop in lesson scripts): ``make_robert_pool_ranking_fn``,
``make_choose_batch_from_ranking_fn``, ``plot_iteration_diagnostics``, ``print_iteration_batch_summary``,
``update_train_and_pool``, ``print_pool_bo_campaign_recap``, ``print_bo_history_tables_and_plots``,
``print_bh_dataset_optimality_gap``, ``build_unique_feature_universe``, ``reveal_yield_manual_from_prediction``,
``make_reveal_yield_from_dataset``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Light ROBERT hyperparameters for interactive BO loops (see ROBERT tests/test_7api.py).
ROBERT_BO_FAST_KW: dict[str, Any] = {
    "model": ["NN"],
    "n_iter": 1,
    "init_points": 10,
    "repeat_kfolds": 1,
    "kfold": 2,
    "pfi_epochs": 1,
}

# When ``bo_surrogate="robert"`` and ``robert_model_kwargs`` does not set ``workdir``,
# each fit uses a fresh folder under ``Path.cwd() / ROBERT_BO_OUTPUT_PARENT_NAME``.
# ``RobertModel`` runs with that directory as cwd, so CURATE/GENERATE/PREDICT plots
# and CSVs land there (see ROBERT ``robert/api.py``).
ROBERT_BO_OUTPUT_PARENT_NAME = "robert_bo_outputs"

# --- BH pool BO lesson (BH.py): tune runs here --------------------------------
BH_POOL_BO_RANDOM_STATE: int = 7
BH_POOL_BO_BATCH_SIZE: int = 10
BH_POOL_BO_STEPS_DEFAULT: int = 1
BH_POOL_BO_ACQUISITION: str = "ts"  # "ucb" | "ei" | "ts" | "random"
BH_POOL_BO_UCB_BETA: float = 3.5
BH_POOL_BO_EI_XI: float = 0.01
BH_POOL_BO_INIT_STRATEGY: Literal["worst", "random"] = "worst"
BH_POOL_BO_N_INITIAL_POINTS: int = 10
BH_POOL_BO_MIN_NONZERO_LOW_POINTS: int = 5

# --- START pool BO lesson (pool_bo_beginner.py): tune runs here ---------------
START_POOL_BO_RANDOM_STATE: int = 7
START_POOL_BO_BATCH_SIZE: int = 1
START_POOL_BO_STEPS_DEFAULT: int = 8
START_POOL_BO_ACQUISITION: str = "ei"  # "ucb" | "ei" | "ts" | "random"
START_POOL_BO_UCB_BETA: float = 3.5
START_POOL_BO_EI_XI: float = 0.01
START_POOL_BO_TEMPERATURE_SCAN: np.ndarray = np.arange(40, 101, 10)
START_POOL_BO_PD_LOADING_SCAN: np.ndarray = np.array([5, 10, 15])
START_POOL_BO_SELECTFLUOR_SCAN: np.ndarray = np.array([1.0, 1.5, 2.0, 2.5])


def pool_bo_acquisition_display_label(acquisition_mode: str, *, ucb_beta: float) -> str:
    """Human-readable label for pool acquisition scores (used by BH and START lessons)."""
    labels: dict[str, str] = {
        "ucb": f"UCB (beta={ucb_beta})",
        "ei": "EI",
        "ts": "Thompson sampling (independent)",
        "random": "Random (uniform scores)",
    }
    return labels.get(acquisition_mode, acquisition_mode)


def bh_pool_bo_n_steps() -> int:
    """BO iterations for the BH lesson; override with env ``BH_POOL_BO_STEPS``."""
    return int(os.environ.get("BH_POOL_BO_STEPS", str(BH_POOL_BO_STEPS_DEFAULT)))


def bh_pool_bo_acquisition_label(acquisition_mode: str) -> str:
    """Human-readable acquisition label for the BH lesson (uses ``BH_POOL_BO_UCB_BETA`` for UCB)."""
    return pool_bo_acquisition_display_label(
        acquisition_mode, ucb_beta=BH_POOL_BO_UCB_BETA
    )


def start_pool_bo_n_steps() -> int:
    """BO iterations for the START lesson; override with env ``START_POOL_BO_STEPS``."""
    return int(os.environ.get("START_POOL_BO_STEPS", str(START_POOL_BO_STEPS_DEFAULT)))


def start_pool_bo_acquisition_label(acquisition_mode: str) -> str:
    """Human-readable acquisition label for the START lesson (uses ``START_POOL_BO_UCB_BETA`` for UCB)."""
    return pool_bo_acquisition_display_label(
        acquisition_mode, ucb_beta=START_POOL_BO_UCB_BETA
    )


def resolve_repo_file(name: str, base_dir: Path) -> Path:
    """Resolve ``name`` under ``base_dir`` (e.g. ``Path(__file__).resolve().parent`` in a script)."""
    return base_dir / name


def print_banner(title: str) -> None:
    """Pretty section separator."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _set_bo_step_axis_integer_ticks() -> None:
    """BO steps are discrete integers; avoid fractional x tick labels (e.g. 1.5)."""
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


def show_current_figure() -> None:
    """Show then close figure to avoid many-open-figure warnings."""
    plt.tight_layout()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        )
        plt.show()
    plt.close()


def gaussian_process_pipeline(
    cat_cols: list[str], num_cols: list[str], *, random_state: int
) -> Pipeline:
    """Pipeline with preprocessing + Gaussian Process."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        length_scale=1.0, length_scale_bounds=(0.2, 5.0), nu=1.5
    ) + WhiteKernel(noise_level=0.2, noise_level_bounds=(0.01, 2.0))

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-3,
        random_state=random_state,
        n_restarts_optimizer=6,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def random_forest_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    n_estimators: int = 50,
) -> Pipeline:
    """Pipeline with preprocessing + Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def quick_inspect_csv(df: pd.DataFrame, *, target_col: str) -> None:
    """Print a short, beginner-friendly snapshot of the CSV."""
    print_banner("STEP 1A - QUICK CSV INSPECTION")
    n_rows, n_cols = df.shape
    print(f"Rows: {n_rows} | Columns: {n_cols}")

    print("\nFirst 5 rows:")
    print(df.head(5))

    print("\nMissing values per column:")
    print(df.isna().sum())

    if target_col in df.columns:
        print(f"\n{target_col} summary:")
        print(df[target_col].describe())

    object_cols = list(df.select_dtypes(include=["object"]).columns)
    if object_cols:
        print("\nUnique value counts for categorical columns:")
        for col in object_cols:
            print(f"  - {col}: {df[col].nunique()}")


def rf_bootstrap_predict_mean_std(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pool: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit two bootstrapped RF models and combine all trees for uncertainty.

    This keeps RF uncertainty simple for beginners:
    - train RF #1 on bootstrap sample #1
    - train RF #2 on bootstrap sample #2
    - collect predictions from every tree in both forests
    - use tree-level mean and std as prediction + uncertainty proxy
    """
    n_rows = len(x_train)
    rng = np.random.RandomState(random_state)
    all_tree_predictions = []

    for bootstrap_id in range(2):
        sampled_idx = rng.choice(n_rows, size=n_rows, replace=True)
        x_boot = x_train.iloc[sampled_idx].reset_index(drop=True)
        y_boot = y_train.iloc[sampled_idx].reset_index(drop=True)

        rf_pipe = random_forest_pipeline(cat_cols, num_cols, random_state=random_state)
        rf_pipe.fit(x_boot, y_boot)

        x_pool_encoded = rf_pipe.named_steps["prep"].transform(x_pool)
        rf_model = rf_pipe.named_steps["model"]
        for tree in rf_model.estimators_:
            tree_predictions = tree.predict(x_pool_encoded)
            all_tree_predictions.append(tree_predictions)

        print(
            f"[RF diagnostics] finished bootstrap forest {bootstrap_id + 1}/2 "
            f"with {len(rf_model.estimators_)} trees"
        )

    tree_pred_matrix = np.vstack(all_tree_predictions)
    mean_pred = tree_pred_matrix.mean(axis=0)
    std_pred = tree_pred_matrix.std(axis=0, ddof=1)
    return mean_pred, std_pred


def fit_predict_with_uncertainty(
    model_name: str,
    pipe: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pool: pd.DataFrame,
    *,
    random_state: int,
    print_diagnostics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit model and return mean + uncertainty on pool.

    For GPR:
      - uncertainty is predictive standard deviation from GP.
    For RF:
      - uncertainty proxy is std across trees from two bootstrapped forests.
    """
    if model_name == "gpr":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe.fit(x_train, y_train)
        mean_pred, std_pred = pipe.predict(x_pool, return_std=True)
        if print_diagnostics:
            gp_model = pipe.named_steps["model"]
            print(
                "[GPR diagnostics] "
                f"kernel={gp_model.kernel_} | "
                f"mu_std={np.std(mean_pred):.3f} | "
                f"mu_range=({np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}) | "
                f"sd_range=({np.min(std_pred):.3f}, {np.max(std_pred):.3f})"
            )
        return mean_pred, std_pred

    if model_name == "rf":
        preprocessor = pipe.named_steps["prep"]
        transformers = preprocessor.transformers
        cat_cols_rf = list(transformers[0][2])
        num_cols_rf = list(transformers[1][2])
        mean_pred, std_pred = rf_bootstrap_predict_mean_std(
            x_train=x_train,
            y_train=y_train,
            x_pool=x_pool,
            cat_cols=cat_cols_rf,
            num_cols=num_cols_rf,
            random_state=random_state,
        )
        return mean_pred, std_pred

    raise ValueError(f"Unknown model_name={model_name}")


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.01
) -> np.ndarray:
    """Expected improvement for maximization."""
    sigma = np.clip(sigma, 1e-12, None)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma <= 1e-12, 0.0, ei)


def upper_confidence_bound(
    mu: np.ndarray, sigma: np.ndarray, beta: float
) -> np.ndarray:
    """UCB for maximization: mean + beta * uncertainty."""
    sigma = np.clip(sigma, 1e-12, None)
    return mu + beta * sigma


def thompson_scores(
    mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Independent Thompson scores (marginal GP/RF approximation): sample N(mu, sigma^2) per arm."""
    sigma = np.clip(sigma, 1e-12, None)
    return rng.normal(loc=mu, scale=sigma)


def acquisition_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    *,
    mode: str,
    rng: np.random.Generator,
    ucb_beta: float,
    ei_xi: float,
) -> np.ndarray:
    """Dispatch acquisition rule (ucb | ei | ts | random)."""
    if mode == "ucb":
        return upper_confidence_bound(mu, sigma, ucb_beta)
    if mode == "ei":
        return expected_improvement(mu, sigma, best_y, xi=ei_xi)
    if mode == "ts":
        return thompson_scores(mu, sigma, rng)
    if mode == "random":
        return rng.random(mu.shape[0])
    raise ValueError(
        f"Unknown ACQUISITION={mode!r}; use 'ucb', 'ei', 'ts', or 'random'."
    )


def plot_iteration_diagnostics(
    ranking: pd.DataFrame,
    selected_batch: pd.DataFrame,
    feature_cols: list[str],
    step: int,
    best_so_far: float,
    batch_size: int,
    *,
    score_col: str,
    acquisition_label: str,
    selected_legend_suffix: str | None = None,
) -> None:
    """Visualize BO state for one batch iteration."""
    if score_col not in ranking.columns:
        raise KeyError(f"ranking is missing score column {score_col!r}")
    selected_keys = set(map(tuple, selected_batch[feature_cols].to_numpy()))
    ranking_plot = ranking.copy()
    ranking_plot["is_selected_batch"] = (
        ranking_plot[feature_cols].apply(tuple, axis=1).isin(selected_keys)
    )

    legend_selected = (
        selected_legend_suffix
        if selected_legend_suffix is not None
        else f"Selected batch (top {batch_size} by {acquisition_label})"
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        ranking_plot.loc[~ranking_plot["is_selected_batch"], "pred_std"],
        ranking_plot.loc[~ranking_plot["is_selected_batch"], "pred_mean"],
        c=ranking_plot.loc[~ranking_plot["is_selected_batch"], score_col],
        cmap="viridis",
        alpha=0.45,
        s=28,
    )
    plt.scatter(
        ranking_plot.loc[ranking_plot["is_selected_batch"], "pred_std"],
        ranking_plot.loc[ranking_plot["is_selected_batch"], "pred_mean"],
        color="red",
        edgecolor="black",
        s=75,
        zorder=5,
        label=legend_selected,
    )
    plt.colorbar(sc, label=acquisition_label)
    plt.axhline(
        best_so_far,
        linestyle="--",
        linewidth=1,
        color="gray",
        label="Current best observed",
    )
    plt.xlabel("Predicted uncertainty (std)")
    plt.ylabel("Predicted mean yield")
    plt.title(f"Iteration {step}: uncertainty vs predicted mean (batch-highlighted)")
    plt.legend()
    show_current_figure()

    top_k = ranking.head(max(20, batch_size)).copy()
    top_k_keys = top_k[feature_cols].apply(tuple, axis=1)
    top_k_selected = top_k_keys.isin(selected_keys)
    labels = []
    label_cols = feature_cols[:4]
    for _, row in top_k.iterrows():
        label_parts = [f"{col}={row[col]}" for col in label_cols]
        labels.append(" | ".join(label_parts))
    bar_colors = ["tomato" if is_sel else "steelblue" for is_sel in top_k_selected]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(top_k)), top_k[score_col], color=bar_colors)
    plt.xticks(range(len(top_k)), labels, rotation=45, ha="right")
    plt.ylabel(acquisition_label)
    plt.xlabel(f"Top candidates (highest {acquisition_label} on left)")
    plt.title(
        f"Iteration {step}: top candidates by {acquisition_label} (selected batch in red)"
    )
    show_current_figure()

    batch_preview_cols = feature_cols + ["pred_mean", "pred_std", score_col]
    selected_preview = selected_batch[batch_preview_cols].copy().reset_index(drop=True)
    selected_preview.insert(0, "batch_rank", np.arange(1, len(selected_preview) + 1))
    print(f"[Plot note] Selected batch for iteration {step}:")
    print(selected_preview)


# --- Session layout / imports -------------------------------------------------


def session_root_with_bo_common(start: Path) -> Path:
    """Return directory containing ``bo_common.py`` (Session_7_Bayesian_Opt)."""
    here = start.resolve()
    if (here / "bo_common.py").exists():
        return here
    parent = here.parent
    if (parent / "bo_common.py").exists():
        return parent
    raise ImportError(
        "Cannot find bo_common.py. Run with cwd = Session_7_Bayesian_Opt, BH/, or START/, "
        "or run the lesson script from its folder."
    )


def ensure_session_importable(start: Path) -> Path:
    """Insert session root on ``sys.path`` if needed; return that root."""
    root = session_root_with_bo_common(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_lesson_data_dir(csv_basename: str, *, script_parent: Path | None) -> Path:
    """
    Directory containing the lesson CSV.

    Pass ``script_parent=Path(__file__).resolve().parent`` from scripts.
    In notebooks (``script_parent is None``), try cwd and ``cwd/BH`` / ``cwd/START``.
    """
    if script_parent is not None:
        return script_parent.resolve()
    cwd = Path.cwd().resolve()
    if (cwd / csv_basename).exists():
        return cwd
    for sub in ("BH", "START"):
        if (cwd / sub / csv_basename).exists():
            return cwd / sub
    return cwd


def print_full_dataframe_inspection(df: pd.DataFrame) -> None:
    """STEP 1B-style snapshot: shape, dtypes, head, NA counts, describe."""
    print_banner("STEP 1B - FULL DATA INSPECTION")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumerical summary:")
    print(df.describe(include=[np.number]))
    print("\nCategorical summary:")
    print(df.describe(include=["object"]))


def prompt_measured_yield(
    chosen_features: pd.DataFrame,
    step: int,
    batch_rank: int,
    suggested_yield: float,
) -> float:
    """
    Ask the user for an observed yield for the selected condition.

    Strict manual mode: input is required; empty responses are rejected.
    """
    print("\nSelected condition for lab execution:")
    print(chosen_features)
    print(
        f"Model-suggested yield is {suggested_yield:.3f}. "
        "Enter your measured yield (required; no default accepted)."
    )

    while True:
        user_text = input(
            f"Step {step}, batch rank {batch_rank} measured yield: "
        ).strip()
        if user_text == "":
            print("Input is required. Please enter a numeric measured yield.")
            continue
        try:
            measured_value = float(user_text)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if measured_value < 0 or measured_value > 100:
            print("Yield is usually between 0 and 100. Please re-enter.")
            continue
        return measured_value


def validate_bo_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
) -> None:
    """
    Check that ``FEATURE_COLS`` / ``CAT_COLS`` / ``NUM_COLS`` match the loaded table.

    ``CAT_COLS`` and ``NUM_COLS`` must partition ``FEATURE_COLS`` exactly (no overlap, no gaps).
    """
    fc = list(feature_cols)
    if len(fc) != len(set(fc)):
        raise ValueError("FEATURE_COLS contains duplicate names.")
    cat_set = set(cat_cols)
    num_set = set(num_cols)
    overlap = cat_set & num_set
    if overlap:
        raise ValueError(f"Columns listed as both categorical and numeric: {sorted(overlap)}")
    if cat_set | num_set != set(fc):
        only_feat = set(fc) - cat_set - num_set
        extra = (cat_set | num_set) - set(fc)
        raise ValueError(
            "CAT_COLS and NUM_COLS must list every FEATURE_COLS name exactly once. "
            f"In FEATURE_COLS only (missing from cat/num): {sorted(only_feat)}; "
            f"in cat/num but not in FEATURE_COLS: {sorted(extra)}."
        )
    unknown = [c for c in fc + [target_col] if c not in df.columns]
    if unknown:
        raise ValueError(f"Column(s) not found in dataframe: {unknown}")


# --- BH lesson: seed, lookup, random baseline ---------------------------------


def build_initial_low_yield_seed(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    n_initial_low_points: int,
    min_nonzero_low_points: int = BH_POOL_BO_MIN_NONZERO_LOW_POINTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build initial observed set from low-yield rows.

    Rule: include at least ``min_nonzero_low_points`` from the lowest non-zero yields,
    then fill remaining slots using the globally lowest yields (zeros allowed).
    """
    if n_initial_low_points <= 0:
        raise ValueError("n_initial_low_points must be > 0.")
    if n_initial_low_points < min_nonzero_low_points:
        raise ValueError("n_initial_low_points must be >= min_nonzero_low_points.")

    df_sorted = df.sort_values(target_col, ascending=True).reset_index(drop=True)
    nonzero_low = df_sorted.loc[df_sorted[target_col] > 0].head(min_nonzero_low_points)
    if len(nonzero_low) < min_nonzero_low_points:
        raise ValueError(
            f"Need at least {min_nonzero_low_points} non-zero rows, found {len(nonzero_low)}."
        )

    seed_df = nonzero_low.copy()
    remaining_n = n_initial_low_points - len(seed_df)
    if remaining_n > 0:
        remaining = df_sorted.drop(index=seed_df.index, errors="ignore")
        extra = remaining.head(remaining_n)
        seed_df = pd.concat([seed_df, extra], ignore_index=True)

    seed_df = seed_df.drop_duplicates(subset=feature_cols, keep="first").reset_index(
        drop=True
    )
    if len(seed_df) < n_initial_low_points:
        already = set(map(tuple, seed_df[feature_cols].to_numpy()))
        candidates = df_sorted.loc[
            ~df_sorted[feature_cols].apply(tuple, axis=1).isin(already)
        ].head(n_initial_low_points - len(seed_df))
        seed_df = pd.concat([seed_df, candidates], ignore_index=True)

    seed_df = seed_df.head(n_initial_low_points).reset_index(drop=True)
    seed_keys = set(map(tuple, seed_df[feature_cols].to_numpy()))
    candidate_df = df.loc[
        ~df[feature_cols].apply(tuple, axis=1).isin(seed_keys), feature_cols
    ].copy()
    return seed_df, candidate_df.reset_index(drop=True)


def _print_bo_initial_observed_summary(
    strategy: Literal["worst", "random"],
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    target_col: str,
    *,
    min_nonzero_low_points: int,
) -> None:
    """Console summary for ``build_bo_initial_observed_and_pool`` (worst-seed diagnostics stay here)."""
    print(f"Initial train/pool split strategy: {strategy!r}")
    n_nonzero = int((observed_df[target_col] > 0).sum())
    print(
        f"Initial training set size: {len(observed_df)} "
        f"(non-zero yields in initial set: {n_nonzero})"
    )
    if strategy == "worst":
        ok = n_nonzero >= min_nonzero_low_points
        print(
            f"Constraint check: at least {min_nonzero_low_points} non-zero low-yield points => {ok}"
        )
        print(
            "\nPedagogy note: we intentionally start from poor outcomes so the BO trajectory "
            "shows how uncertainty-aware search can recover toward better conditions."
        )
    else:
        print(
            "\nPedagogy note: random initialization samples arbitrary conditions; compare BO recovery "
            "against the worst-case seed when switching strategy to worst."
        )
    print(f"Remaining candidates available for BO: {len(candidate_df)}")


def build_bo_initial_observed_and_pool(
    universe_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    strategy: Literal["worst", "random"] = BH_POOL_BO_INIT_STRATEGY,
    n_points: int,
    random_state: int,
    min_nonzero_low_points: int = BH_POOL_BO_MIN_NONZERO_LOW_POINTS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the discrete universe into initial training rows and the remaining candidate pool.

    Defaults for ``strategy`` and ``min_nonzero_low_points`` follow ``BH_POOL_BO_INIT_STRATEGY`` and
    ``BH_POOL_BO_MIN_NONZERO_LOW_POINTS``.

    ``strategy="worst"`` delegates to ``build_initial_low_yield_seed`` (deterministic); ``random_state`` is unused.
    ``strategy="random"`` samples ``n_points`` universe rows without replacement using
    ``numpy.random.default_rng(random_state)``; ``min_nonzero_low_points`` is ignored in that case.

    When ``verbose`` is true, prints strategy-specific diagnostics (including worst-seed constraint checks).
    """
    if strategy == "worst":
        observed_df, candidate_df = build_initial_low_yield_seed(
            universe_df,
            feature_cols,
            target_col,
            n_initial_low_points=n_points,
            min_nonzero_low_points=min_nonzero_low_points,
        )
    elif strategy == "random":
        if n_points <= 0:
            raise ValueError("n_points must be > 0.")
        u_n = len(universe_df)
        if n_points > u_n:
            raise ValueError(f"n_points ({n_points}) exceeds universe size ({u_n}).")
        rng = np.random.default_rng(random_state)
        idx = rng.choice(u_n, size=n_points, replace=False)
        observed_df = universe_df.iloc[idx].reset_index(drop=True)
        seed_keys = set(map(tuple, observed_df[feature_cols].to_numpy()))
        candidate_df = universe_df.loc[
            ~universe_df[feature_cols].apply(tuple, axis=1).isin(seed_keys),
            feature_cols,
        ].copy()
        candidate_df = candidate_df.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown strategy={strategy!r}; use 'worst' or 'random'.")

    if verbose:
        _print_bo_initial_observed_summary(
            strategy,
            observed_df,
            candidate_df,
            target_col,
            min_nonzero_low_points=min_nonzero_low_points,
        )
    return observed_df, candidate_df


def lookup_true_yield(
    all_data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    chosen_features: pd.DataFrame,
) -> float:
    """Reveal the true objective value from a dataset for the chosen condition."""
    chosen_key = tuple(chosen_features.iloc[0][feature_cols].to_list())
    all_keys = all_data[feature_cols].apply(tuple, axis=1)
    matches = all_data.loc[all_keys == chosen_key, target_col]
    if matches.empty:
        raise ValueError(f"Chosen condition not found in dataset: {chosen_key}")
    return float(matches.iloc[0])


def build_unique_feature_universe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    head_rows: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One row per unique reaction condition in the HTE table (discrete search space for pool BO).

    We keep the first occurrence of each feature tuple so each point has a single reference yield.
    """
    universe_df = (
        df[feature_cols + [target_col]]
        .drop_duplicates(subset=feature_cols, keep="first")
        .reset_index(drop=True)
    )
    if verbose:
        print(f"Universe built from dataset rows (first {head_rows}):")
        print(universe_df.head(head_rows))
        print(f"Unique feature combinations in universe: {len(universe_df)}")
    return universe_df


def random_baseline_best_yield(
    universe_df: pd.DataFrame,
    initial_pool: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed_best: float,
    n_picks: int,
    rng: np.random.Generator,
) -> float:
    """Max yield after ``n_picks`` uniform random draws without replacement from ``initial_pool``."""
    pool_n = len(initial_pool)
    if pool_n == 0 or n_picks <= 0:
        return float(seed_best)
    take = min(n_picks, pool_n)
    idx = rng.choice(pool_n, size=take, replace=False)
    picked = initial_pool.iloc[idx].reset_index(drop=True)
    ys: list[float] = []
    for i in range(len(picked)):
        feats = picked.iloc[[i]][feature_cols]
        ys.append(lookup_true_yield(universe_df, feature_cols, target_col, feats))
    return max(float(seed_best), max(ys) if ys else float(seed_best))


def random_baseline_distribution(
    universe_df: pd.DataFrame,
    initial_pool: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed_best: float,
    n_picks: int,
    rng: np.random.Generator,
    n_repeats: int,
) -> tuple[float, float, np.ndarray]:
    """Monte Carlo distribution of best yields from repeated random experiments."""
    samples = np.empty(n_repeats, dtype=float)
    for i in range(n_repeats):
        samples[i] = random_baseline_best_yield(
            universe_df,
            initial_pool,
            feature_cols,
            target_col,
            seed_best,
            n_picks,
            np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
    return float(np.median(samples)), float(np.mean(samples)), samples


# --- START lesson: scanned pool -----------------------------------------------


def build_scanned_pool_with_observed_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    *,
    temperature_scan: np.ndarray,
    pd_loading_scan: np.ndarray,
    selectfluor_scan: np.ndarray,
    random_state: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cross-join unique categorical rows with a numeric scan grid; flag observed rows.

    Returns ``(pool_with_flag, observed_df, candidate_df)`` where ``observed_df`` is ``df.copy()``.
    """
    if verbose:
        print_banner("STEP 3 - BUILD CANDIDATE POOL (OBSERVED + SCANNED CONDITIONS)")
    unique_cat = df[cat_cols].drop_duplicates().reset_index(drop=True)
    if verbose:
        print("Unique categorical combinations (first 10):")
        print(unique_cat.head(10))
        print(f"Count of unique categorical combinations: {len(unique_cat)}")

    numeric_grid = pd.MultiIndex.from_product(
        [temperature_scan, pd_loading_scan, selectfluor_scan],
        names=["Temperature", "Pd_loading", "Selectfluor"],
    ).to_frame(index=False)
    if verbose:
        print("\nNumeric scan grid (first 10):")
        print(numeric_grid.head(10))
        print(f"Count of numeric grid rows: {len(numeric_grid)}")

    pool = unique_cat.merge(numeric_grid, how="cross")
    pool = pool[feature_cols].copy()
    if verbose:
        print("\nCandidate pool from cross join (first 10):")
        print(pool.head(10))
        print(f"Candidate pool size before dedup: {len(pool)}")
    pool = pool.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"Candidate pool size after dedup: {len(pool)}")

    observed_features = df[feature_cols].drop_duplicates()
    observed_keys = set(map(tuple, observed_features.to_numpy()))
    pool["_is_observed"] = [
        tuple(row) in observed_keys for row in pool[feature_cols].to_numpy()
    ]
    if verbose:
        print("\nObserved-vs-unobserved count in pool:")
        print(pool["_is_observed"].value_counts(dropna=False))
        print("\nRandom pool sample:")
        print(pool.sample(min(10, len(pool)), random_state=random_state))

    observed_df = df.copy()
    candidate_df = pool.loc[~pool["_is_observed"], feature_cols].copy()
    if verbose:
        print_banner("STEP 4 - TRAINING DATA AND AVAILABLE (UNOBSERVED) CANDIDATES")
        print(f"Observed experiments available for training: {len(observed_df)}")
        print(f"Unobserved candidates available for BO: {len(candidate_df)}")
        print("\nObserved training rows (head):")
        print(observed_df.head())
        print("\nCandidate rows (head):")
        print(candidate_df.head())

    return pool, observed_df, candidate_df


# --- GPR-only diagnostics (optional before BO loop) ---------------------------


def fit_gpr_pool_diagnostic_table(
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
) -> pd.DataFrame:
    """Fit GPR on observed data; return pool rows with predicted mean and std."""
    x_train = observed_df[feature_cols]
    y_train = observed_df[target_col]
    gpr_pipe = gaussian_process_pipeline(cat_cols, num_cols, random_state=random_state)
    mu, sd = fit_predict_with_uncertainty(
        "gpr", gpr_pipe, x_train, y_train, candidate_df, random_state=random_state
    )
    diagnostic = candidate_df.copy()
    diagnostic["pred_mean"] = mu
    diagnostic["pred_std"] = sd
    return diagnostic


def print_gpr_pool_diagnostic_summary(diagnostic: pd.DataFrame, *, k: int = 5) -> None:
    print("GPR on pool: predicted mean / std summary:")
    print(diagnostic[["pred_mean", "pred_std"]].describe())
    print(f"\nPool rows with highest predicted uncertainty (top {k}):")
    print(diagnostic.nlargest(k, "pred_std"))


def plot_gpr_pool_mean_vs_std(diagnostic: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(diagnostic["pred_mean"], diagnostic["pred_std"], alpha=0.5)
    plt.xlabel("GPR predicted mean (yield)")
    plt.ylabel("GPR predictive std")
    plt.title("GPR pool diagnostic: mean vs uncertainty")
    show_current_figure()


def _cv_n_splits_effective(
    n_obs: int, *, n_splits: int | None, adaptive_splits: bool
) -> int:
    if adaptive_splits and n_splits is None:
        return 3 if n_obs < 20 else min(5, max(2, n_obs // 2))
    if n_splits is None:
        return 5
    return n_splits


def run_gpr_cv_diagnostics(
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    n_splits: int | None = None,
    adaptive_splits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    K-fold CV for GPR on observed rows: per-fold metrics, uncertainty–error check, two figures.

    Use ``adaptive_splits=True`` and ``n_splits=None`` for BH-style fold count;
    use ``n_splits=5, adaptive_splits=False`` for fixed 5-fold (START).
    """
    n_obs_cv = len(observed_df)
    if n_obs_cv < 2:
        print_banner("STEP 5B - K-FOLD CV (GPR)")
        print("Skipping CV: need at least 2 observed rows.")
        return pd.DataFrame(), pd.DataFrame()

    cv_n_splits = _cv_n_splits_effective(
        n_obs_cv, n_splits=n_splits, adaptive_splits=adaptive_splits
    )
    cv_n_splits = min(cv_n_splits, n_obs_cv)
    if cv_n_splits < 2:
        print_banner("STEP 5B - K-FOLD CV (GPR)")
        print("Skipping CV: not enough rows for two folds.")
        return pd.DataFrame(), pd.DataFrame()

    print_banner(f"STEP 5B - {cv_n_splits}-FOLD CV: GPR TEST PREDICTIONS")
    print(
        f"CV folds={cv_n_splits} (n_obs={n_obs_cv}; fewer folds when the initial seed set is small)."
    )
    kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    y_std_all: list[float] = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(observed_df), start=1):
        train_fold = observed_df.iloc[train_idx].reset_index(drop=True)
        test_fold = observed_df.iloc[test_idx].reset_index(drop=True)
        x_train_fold = train_fold[feature_cols]
        y_train_fold = train_fold[target_col]
        x_test_fold = test_fold[feature_cols]
        y_test_fold = test_fold[target_col].to_numpy(dtype=float)

        gpr_fold = gaussian_process_pipeline(
            cat_cols, num_cols, random_state=random_state
        )
        pred_fold, std_fold = fit_predict_with_uncertainty(
            "gpr",
            gpr_fold,
            x_train_fold,
            y_train_fold,
            x_test_fold,
            random_state=random_state,
            print_diagnostics=False,
        )
        pred_fold = pred_fold.astype(float)
        std_fold = std_fold.astype(float)

        y_true_all.extend(y_test_fold.tolist())
        y_pred_all.extend(pred_fold.tolist())
        y_std_all.extend(std_fold.tolist())

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test": len(test_fold),
                "mae": mean_absolute_error(y_test_fold, pred_fold),
                "r2": r2_score(y_test_fold, pred_fold),
            }
        )

    cv_df = pd.DataFrame(fold_rows)
    print("Per-fold test metrics (GPR):")
    print(cv_df)
    print("\nMean CV metrics:")
    print(cv_df[["mae", "r2"]].mean().to_frame("mean"))

    cv_pred_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true_all, dtype=float),
            "y_pred": np.asarray(y_pred_all, dtype=float),
            "std": np.asarray(y_std_all, dtype=float),
        }
    )
    cv_pred_df["abs_err"] = np.abs(cv_pred_df["y_true"] - cv_pred_df["y_pred"])

    pearson_r, pearson_p = pearsonr(cv_pred_df["std"], cv_pred_df["abs_err"])
    spearman_rho, spearman_p = spearmanr(cv_pred_df["std"], cv_pred_df["abs_err"])
    print("\nUncertainty vs absolute error on held-out folds (GPR predictive std):")
    print(
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
        f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.2e})"
    )

    cv_pred_df["uncertainty_bin"] = pd.qcut(cv_pred_df["std"], q=4, duplicates="drop")
    bin_summary = (
        cv_pred_df.groupby("uncertainty_bin", observed=True)
        .agg(
            n_points=("abs_err", "size"),
            mean_pred_std=("std", "mean"),
            mean_abs_err=("abs_err", "mean"),
        )
        .reset_index()
    )
    print(
        "\nGPR uncertainty-bin summary (higher mean_pred_std should trend to higher mean_abs_err):"
    )
    print(bin_summary)

    plt.figure(figsize=(8, 6))
    plt.scatter(cv_pred_df["y_true"], cv_pred_df["y_pred"], alpha=0.7)
    low = min(cv_pred_df["y_true"].min(), cv_pred_df["y_pred"].min())
    high = max(cv_pred_df["y_true"].max(), cv_pred_df["y_pred"].max())
    plt.plot([low, high], [low, high], "--", linewidth=1, color="black")
    plt.xlabel("True yield (test folds)")
    plt.ylabel("GPR predicted yield")
    plt.title(f"{cv_n_splits}-fold CV: GPR predictions vs held-out truth")
    show_current_figure()

    plt.figure(figsize=(8, 6))
    plt.scatter(cv_pred_df["std"], cv_pred_df["abs_err"], alpha=0.7)
    plt.xlabel("GPR predictive std (test folds)")
    plt.ylabel("Absolute error (test folds)")
    plt.title(f"{cv_n_splits}-fold CV: does uncertainty track error?")
    show_current_figure()

    return cv_df, cv_pred_df


def run_start_lesson_gpr_diagnostics(
    observed_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    n_splits: int | None = 5,
    adaptive_splits: bool = False,
) -> None:
    """
    Optional block before BO: one GPR fit on observed data (pool summary + plot), then GPR k-fold CV.
    """
    print_banner("STEP 5 - MODEL DIAGNOSTIC: GPR (SINGLE SURROGATE)")
    diagnostic = fit_gpr_pool_diagnostic_table(
        observed_df,
        candidate_df,
        feature_cols,
        target_col,
        cat_cols,
        num_cols,
        random_state=random_state,
    )
    print_gpr_pool_diagnostic_summary(diagnostic)
    plot_gpr_pool_mean_vs_std(diagnostic)

    run_gpr_cv_diagnostics(
        observed_df,
        feature_cols,
        target_col,
        cat_cols,
        num_cols,
        random_state=random_state,
        n_splits=n_splits,
        adaptive_splits=adaptive_splits,
    )


def _prepare_robert_feature_frame(
    x: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    num_cols: list[str],
) -> pd.DataFrame:
    """
    Coerce columns so ROBERT's string-based categorical detection matches explicit ``cat_cols``.

    Numeric columns are coerced with ``pd.to_numeric``; categorical columns are cast to string.
    """
    out = x.loc[:, feature_cols].copy()
    for col in cat_cols:
        out[col] = out[col].astype(str)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _import_robert_model() -> Any:
    try:
        from robert import RobertModel
    except ImportError as err:
        raise ImportError(
            "The `robert` package is required for bo_surrogate='robert' "
            "(install ROBERT, e.g. conda env `cheminf`)."
        ) from err
    return RobertModel


def _robert_align_pool_features(
    model: Any, pool_df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Build a feature frame whose columns match ``model.model_data_['X_descriptors']``.

    Mirrors ROBERT's ``load_db_n_params`` predict path: try direct column subset, then
    ``categorical_transform(..., 'predict')`` on object columns (``RobertModel.predict`` does not).
    """
    from robert.utils import categorical_transform

    descs = list(model.model_data_["X_descriptors"])
    x_raw = pool_df.loc[:, feature_cols].copy()
    try:
        return x_raw[descs]
    except KeyError:
        pass

    class _PredictCatAdapter:
        __slots__ = ("args",)

        def __init__(self) -> None:
            rob_kw = dict(getattr(model, "_rob_kwargs", {}))
            y_key = str(model.y_col_)
            if y_key in x_raw.columns:
                y_key = "__robert_bo_align_y__"
            self.args = SimpleNamespace(
                ignore=list(rob_kw.get("ignore", [])),
                y=y_key,
                categorical=str(rob_kw.get("categorical", "onehot")),
            )

    adapter = _PredictCatAdapter()
    x_tf = categorical_transform(adapter, x_raw.copy(), "predict")
    for col in descs:
        if col not in x_tf.columns:
            x_tf[col] = 0.0
    return x_tf[descs]


# --- Batched BO loop pieces ---------------------------------------------------


def update_train_and_pool(
    current_train: pd.DataFrame,
    current_pool: pd.DataFrame,
    chosen_records: pd.DataFrame,
    chosen_keys: list[tuple],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append chosen batch to train and remove all selected keys from pool."""
    new_train = pd.concat([current_train, chosen_records], ignore_index=True)
    pool_keys_local = current_pool[feature_cols].apply(tuple, axis=1)
    chosen_key_set = set(chosen_keys)
    new_pool = current_pool.loc[~pool_keys_local.isin(chosen_key_set)].reset_index(
        drop=True
    )
    return new_train, new_pool


def rank_pool_by_surrogate(
    current_train: pd.DataFrame,
    current_pool: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    bo_surrogate: str,
    acquisition_mode: str | None,
    score_col: str,
    bo_rng: np.random.Generator,
    ucb_beta: float,
    ei_xi: float,
    robert_model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Fit surrogate on ``current_train``, score ``current_pool``, sort descending.

    If ``acquisition_mode`` is ``None``, use expected improvement with ``ei_xi`` and column ``score_col`` (e.g. ``EI``).
    Otherwise ``acquisition_mode`` is ``ucb`` | ``ei`` | ``ts`` | ``random`` and scores go to ``score_col`` (e.g. ``acq``).
    For ``random``, the surrogate is still fit so ``pred_mean`` / ``pred_std`` are available for plots; ranking order
    is uniform random each call (via ``bo_rng``).
    """
    x_train_local = current_train[feature_cols]
    y_train_local = current_train[target_col]
    if bo_surrogate == "gpr":
        pipe_local = gaussian_process_pipeline(
            cat_cols, num_cols, random_state=random_state
        )
        mu_local, sd_local = fit_predict_with_uncertainty(
            "gpr",
            pipe_local,
            x_train_local,
            y_train_local,
            current_pool,
            random_state=random_state,
        )
    elif bo_surrogate == "rf":
        pipe_local = random_forest_pipeline(
            cat_cols, num_cols, random_state=random_state
        )
        mu_local, sd_local = fit_predict_with_uncertainty(
            "rf",
            pipe_local,
            x_train_local,
            y_train_local,
            current_pool,
            random_state=random_state,
        )
    elif bo_surrogate == "robert":
        fitted_local = fit_bo_surrogate_on_train(
            current_train,
            feature_cols,
            target_col,
            cat_cols,
            num_cols,
            random_state=random_state,
            bo_surrogate="robert",
            robert_model_kwargs=robert_model_kwargs,
        )
        mu_local, sd_local = predict_bo_surrogate_on_pool(fitted_local, current_pool)
    else:
        raise ValueError(
            f"Unknown bo_surrogate={bo_surrogate!r}; use 'gpr', 'rf', or 'robert'."
        )

    best_local = float(y_train_local.max())
    if acquisition_mode is None:
        scores = expected_improvement(mu_local, sd_local, best_local, xi=ei_xi)
    else:
        scores = acquisition_scores(
            mu_local,
            sd_local,
            best_local,
            mode=acquisition_mode,
            rng=bo_rng,
            ucb_beta=ucb_beta,
            ei_xi=ei_xi,
        )

    ranking_local = current_pool.copy()
    ranking_local["pred_mean"] = mu_local
    ranking_local["pred_std"] = sd_local
    ranking_local[score_col] = scores
    ranking_local = ranking_local.sort_values(score_col, ascending=False).reset_index(
        drop=True
    )
    return ranking_local, best_local


class FittedBOSurrogate(NamedTuple):
    """Result of ``fit_bo_surrogate_on_train``: sklearn ``Pipeline`` or ``RobertModel`` plus training arrays."""

    model_name: str
    pipeline: Any
    x_train: pd.DataFrame
    y_train: pd.Series
    cat_cols: list[str]
    num_cols: list[str]
    random_state: int


def fit_bo_surrogate_on_train(
    current_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    bo_surrogate: str = "gpr",
    robert_model_kwargs: dict[str, Any] | None = None,
) -> FittedBOSurrogate:
    """
    Fit preprocessing + surrogate on ``current_train`` (features + target column only).

    For GPR this is the full fit used later by ``predict_bo_surrogate_on_pool``.
    For RF the pipeline is fitted for inspection; pool mean/std still use the bootstrap
    path inside ``predict_bo_surrogate_on_pool`` (same behavior as before).
    For ROBERT, ``pipeline`` holds a fitted ``RobertModel`` (fresh workdir per fit).
    ``robert_model_kwargs`` is merged into the ``RobertModel`` constructor after
    ``ROBERT_BO_FAST_KW`` (lesson defaults); use it for documented ROBERT options such as
    ``categorical`` (``"onehot"`` or ``"numbers"``) or ``ignore``. Pass ``workdir`` there
    to fix outputs to a path; otherwise each fit uses a subfolder of
    ``Path.cwd() / ROBERT_BO_OUTPUT_PARENT_NAME``.
    """
    x_train = current_train[feature_cols]
    y_train = current_train[target_col]
    if bo_surrogate == "gpr":
        pipe = gaussian_process_pipeline(cat_cols, num_cols, random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe.fit(x_train, y_train)
        return FittedBOSurrogate(
            "gpr", pipe, x_train, y_train, cat_cols, num_cols, random_state
        )
    if bo_surrogate == "rf":
        pipe = random_forest_pipeline(cat_cols, num_cols, random_state=random_state)
        pipe.fit(x_train, y_train)
        return FittedBOSurrogate(
            "rf", pipe, x_train, y_train, cat_cols, num_cols, random_state
        )
    if bo_surrogate == "robert":
        RobertModel = _import_robert_model()
        user_robert_kw = dict(robert_model_kwargs or {})
        user_workdir = user_robert_kw.pop("workdir", None)
        if user_workdir is not None:
            workdir = Path(user_workdir).resolve()
        else:
            parent = Path.cwd() / ROBERT_BO_OUTPUT_PARENT_NAME
            parent.mkdir(parents=True, exist_ok=True)
            workdir = Path(
                tempfile.mkdtemp(prefix="session7_bo_robert_", dir=str(parent))
            )
        rob_kw = {
            **ROBERT_BO_FAST_KW,
            **user_robert_kw,
            "command_line": False,
            "csv_test": "",
        }
        model = RobertModel(
            problem_type="reg",
            filter_mode="no_pfi",
            workdir=workdir,
            seed=random_state,
            **rob_kw,
        )
        model._rob_kwargs = dict(rob_kw)
        y_fit = y_train.copy()
        if y_fit.name != target_col:
            y_fit = y_fit.rename(target_col)
        x_train_rob = _prepare_robert_feature_frame(
            x_train, feature_cols, cat_cols, num_cols
        )
        model.fit(x_train_rob, y_fit)
        return FittedBOSurrogate(
            "robert", model, x_train_rob, y_fit, cat_cols, num_cols, random_state
        )
    raise ValueError(
        f"Unknown bo_surrogate={bo_surrogate!r}; use 'gpr', 'rf', or 'robert'."
    )


def predict_bo_surrogate_on_pool(
    fitted: FittedBOSurrogate,
    pool_df: pd.DataFrame,
    *,
    print_diagnostics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and uncertainty on ``pool_df`` using the training fit in ``fitted``.

    GPR: one ``predict(..., return_std=True)`` on the already-fitted pipeline.
    RF: tree-ensemble bootstrap uncertainty (re-fits internally; same as lesson RF path).
    ROBERT: ``RobertModel.predict(..., return_std=True)`` (CV disagreement as std).
    """
    if fitted.model_name == "gpr":
        mean_pred, std_pred = fitted.pipeline.predict(pool_df, return_std=True)
        mean_pred = np.asarray(mean_pred, dtype=float)
        std_pred = np.asarray(std_pred, dtype=float)
        if print_diagnostics:
            gp_model = fitted.pipeline.named_steps["model"]
            print(
                "[GPR diagnostics] "
                f"kernel={gp_model.kernel_} | "
                f"mu_std={np.std(mean_pred):.3f} | "
                f"mu_range=({np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}) | "
                f"sd_range=({np.min(std_pred):.3f}, {np.max(std_pred):.3f})"
            )
        return mean_pred, std_pred

    if fitted.model_name == "rf":
        rf_pipe = random_forest_pipeline(
            fitted.cat_cols, fitted.num_cols, random_state=fitted.random_state
        )
        return fit_predict_with_uncertainty(
            "rf",
            rf_pipe,
            fitted.x_train,
            fitted.y_train,
            pool_df,
            random_state=fitted.random_state,
            print_diagnostics=print_diagnostics,
        )

    if fitted.model_name == "robert":
        model = fitted.pipeline
        feat_cols = list(fitted.x_train.columns)
        x_pool_raw = _prepare_robert_feature_frame(
            pool_df, feat_cols, fitted.cat_cols, fitted.num_cols
        )
        x_pool = _robert_align_pool_features(model, x_pool_raw, feat_cols)
        mean_pred, std_pred = model.predict(x_pool, return_std=True)
        mean_pred = np.asarray(mean_pred, dtype=float)
        std_pred = np.asarray(std_pred, dtype=float)
        if print_diagnostics:
            mcode = str(model.model_data_.get("model", ""))
            print(
                "[ROBERT diagnostics] "
                f"model={mcode} | "
                f"mu_std={np.std(mean_pred):.3f} | "
                f"mu_range=({np.min(mean_pred):.3f}, {np.max(mean_pred):.3f}) | "
                f"sd_range=({np.min(std_pred):.3f}, {np.max(std_pred):.3f})"
            )
        return mean_pred, std_pred

    raise ValueError(f"Unknown model_name={fitted.model_name!r}")


def build_ei_ranked_pool(
    pool_df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    best_y: float,
    *,
    score_col: str = "EI",
    ei_xi: float = 0.01,
) -> tuple[pd.DataFrame, float]:
    """
    Attach predicted mean/std and expected improvement scores; sort pool rows by ``score_col`` descending.
    """
    scores = expected_improvement(mu, sigma, best_y, xi=ei_xi)
    ranking_local = pool_df.copy()
    ranking_local["pred_mean"] = mu
    ranking_local["pred_std"] = sigma
    ranking_local[score_col] = scores
    ranking_local = ranking_local.sort_values(score_col, ascending=False).reset_index(
        drop=True
    )
    return ranking_local, best_y


def rank_pool_by_expected_improvement(
    current_train: pd.DataFrame,
    current_pool: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    score_col: str = "EI",
    ei_xi: float = 0.01,
    bo_surrogate: str = "gpr",
    bo_rng: np.random.Generator | None = None,
    robert_model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Fit surrogate, predict on pool, rank by expected improvement (convenience composition).

    Equivalent to chaining ``fit_bo_surrogate_on_train``, ``predict_bo_surrogate_on_pool``,
    and ``build_ei_ranked_pool``. ``bo_rng`` is ignored (kept for backward-compatible call sites).
    """
    _ = bo_rng
    fitted = fit_bo_surrogate_on_train(
        current_train,
        feature_cols,
        target_col,
        cat_cols,
        num_cols,
        random_state=random_state,
        bo_surrogate=bo_surrogate,
        robert_model_kwargs=robert_model_kwargs,
    )
    mu, sd = predict_bo_surrogate_on_pool(fitted, current_pool)
    best_y = float(fitted.y_train.max())
    return build_ei_ranked_pool(
        current_pool, mu, sd, best_y, score_col=score_col, ei_xi=ei_xi
    )


RevealYield = Callable[[pd.Series, pd.DataFrame, int, int], float]


def reveal_yield_manual_from_prediction(
    row: pd.Series,
    chosen_features: pd.DataFrame,
    batch_rank: int,
    step: int,
) -> float:
    """
    Human-in-the-loop stand-in: ask for a lab yield using the model mean as a soft hint.

    ``row`` supplies ``pred_mean`` from the ranked pool row; learners compare their measurement to the surrogate.
    """
    return prompt_measured_yield(
        chosen_features,
        step,
        batch_rank,
        suggested_yield=float(row["pred_mean"]),
    )


def make_reveal_yield_from_dataset(
    universe_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> RevealYield:
    """
    Return a ``RevealYield`` that looks up the true yield after selection (hidden-until-queried simulation).

    ``row``, ``batch_rank``, and ``step`` are unused but required so the callback matches ``choose_batch_from_ranking``.
    """

    def _reveal(
        _row: pd.Series,
        chosen_features: pd.DataFrame,
        _batch_rank: int,
        _step: int,
    ) -> float:
        return lookup_true_yield(universe_df, feature_cols, target_col, chosen_features)

    return _reveal


def make_robert_pool_ranking_fn(
    feature_cols: list[str],
    target_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    random_state: int,
    acquisition_mode: str,
    score_col: str,
    ucb_beta: float,
    ei_xi: float,
    robert_model_kwargs: dict[str, Any] | None = None,
) -> Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, float]]:
    """
    Bind ROBERT + acquisition settings; lessons assign the result to ``build_ranking_table``.

    Stochastic acquisition (e.g. Thompson sampling, ``acquisition_mode="random"``) uses a single
    ``numpy.random.Generator`` seeded with ``random_state``, advanced across BO iterations.
    """

    bo_rng = np.random.default_rng(random_state)

    def build_ranking_table(
        current_train: pd.DataFrame, current_pool: pd.DataFrame
    ) -> tuple[pd.DataFrame, float]:
        return rank_pool_by_surrogate(
            current_train,
            current_pool,
            feature_cols,
            target_col,
            cat_cols,
            num_cols,
            random_state=random_state,
            bo_surrogate="robert",
            acquisition_mode=acquisition_mode,
            score_col=score_col,
            bo_rng=bo_rng,
            ucb_beta=ucb_beta,
            ei_xi=ei_xi,
            robert_model_kwargs=robert_model_kwargs,
        )

    return build_ranking_table


def make_choose_batch_from_ranking_fn(
    feature_cols: list[str],
    target_col: str,
    score_col: str,
    reveal_yield: RevealYield,
    source_kind: str,
) -> Callable[
    [pd.DataFrame, int, int],
    tuple[pd.DataFrame, list[tuple], list[dict], pd.DataFrame],
]:
    """Bind column names and yield source; lessons assign the result to ``choose_batch_and_record``."""

    def choose_batch_and_record(
        ranking_local: pd.DataFrame, step_local: int, batch_size_local: int
    ) -> tuple[pd.DataFrame, list[tuple], list[dict], pd.DataFrame]:
        return choose_batch_from_ranking(
            ranking_local,
            feature_cols,
            target_col,
            score_col,
            step_local,
            batch_size_local,
            reveal_yield=reveal_yield,
            source_kind=source_kind,
        )

    return choose_batch_and_record


def choose_batch_from_ranking(
    ranking: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    score_col: str,
    step: int,
    batch_size: int,
    *,
    reveal_yield: RevealYield,
    source_kind: str,
) -> tuple[pd.DataFrame, list[tuple], list[dict], pd.DataFrame]:
    """
    Select top ``batch_size`` rows; call ``reveal_yield(row, chosen_features, batch_rank, step)`` per row.

    ``source_kind`` is the suffix in ``bo_step_{step}_batch_{rank}_{source_kind}``.
    """
    batch = ranking.head(batch_size).copy().reset_index(drop=True)
    chosen_records_local: list[pd.DataFrame] = []
    chosen_keys_local: list[tuple] = []
    history_items_local: list[dict] = []

    for batch_rank, (_, chosen_local) in enumerate(batch.iterrows(), start=1):
        chosen_features_local = chosen_local[feature_cols].to_frame().T
        observed_yield_local = reveal_yield(
            chosen_local, chosen_features_local, batch_rank, step
        )

        chosen_record_local = chosen_features_local.copy()
        chosen_record_local[target_col] = observed_yield_local
        chosen_record_local["source"] = (
            f"bo_step_{step}_batch_{batch_rank}_{source_kind}"
        )
        chosen_records_local.append(chosen_record_local)
        chosen_keys_local.append(
            tuple(chosen_features_local.iloc[0][feature_cols].to_list())
        )

        history_item_local = {
            "step": step,
            "batch_rank": batch_rank,
            "best_before": np.nan,
            "pred_mean": float(chosen_local["pred_mean"]),
            "pred_std": float(chosen_local["pred_std"]),
            score_col: float(chosen_local[score_col]),
            target_col: observed_yield_local,
            **{col: chosen_local[col] for col in feature_cols},
        }
        history_items_local.append(history_item_local)

    chosen_records_df = pd.concat(chosen_records_local, ignore_index=True)
    return chosen_records_df, chosen_keys_local, history_items_local, batch


def print_iteration_batch_summary(
    ranking: pd.DataFrame,
    history_items: list[dict],
    target_col: str,
    current_batch_size: int,
) -> None:
    batch_yields = [float(item[target_col]) for item in history_items]
    head = ranking.head(current_batch_size)
    print(
        f"Batch predicted means range: {float(head['pred_mean'].min()):.3f} "
        f"to {float(head['pred_mean'].max()):.3f}"
    )
    print(
        f"Revealed observed yields range: {min(batch_yields):.3f} to {max(batch_yields):.3f}"
    )
    print("Per-candidate prediction vs observation:")
    for item in history_items:
        pred_mean_local = float(item["pred_mean"])
        pred_std_local = float(item["pred_std"])
        obs_local = float(item[target_col])
        print(
            f"  - rank {int(item['batch_rank'])}: "
            f"pred={pred_mean_local:.3f} +/- {pred_std_local:.3f}, "
            f"observed={obs_local:.3f}"
        )


def print_pool_bo_campaign_recap(
    acquisition_label: str,
    batch_size: int,
    *,
    mode: Literal["manual", "dataset"],
) -> None:
    """Short narrative after the BO loop: ``manual`` = lab entry; ``dataset`` = simulated reveal from CSV."""
    if mode == "manual":
        print(
            "\nRecap after BO loop:\n"
            f"- Each iteration ranked the pool with ROBERT using {acquisition_label}.\n"
            f"- Up to {batch_size} point(s) per batch were evaluated (or fewer if the pool was smaller).\n"
            "- New measurements were appended to the training table for the next fit.\n"
        )
    elif mode == "dataset":
        print(
            "\nRecap after BO loop:\n"
            f"- Each iteration ranked the pool with ROBERT using {acquisition_label}.\n"
            f"- Up to {batch_size} point(s) per batch were selected (or fewer if the pool was smaller).\n"
            "- True yields were revealed after selection and appended to training."
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}; use 'manual' or 'dataset'.")


def print_bo_history_tables_and_plots(
    history_df: pd.DataFrame,
    target_col: str,
    score_col: str,
    acquisition_label: str,
    *,
    score_ylabel: str = "Acquisition score",
    empty_iters_hint: str | None = None,
) -> None:
    """STEP 7-style recap: table, key columns, trajectory plots (or a hint when no steps ran)."""
    print_banner("STEP 7 - BO HISTORY TABLES")
    print("Full BO history:")
    print(history_df)
    if len(history_df) > 0:
        print("\nYield progression (best_before vs chosen observed yield):")
        print(history_df[["step", "best_before", target_col, "pred_std", score_col]])
        step_summary = summarize_bo_history_step_table(history_df, target_col, score_col)
        plot_bo_trajectory_figures(
            step_summary,
            acquisition_label=acquisition_label,
            score_ylabel=score_ylabel,
        )
    else:
        msg = empty_iters_hint or (
            "No BO iterations ran; set a positive iteration count in User settings to populate history."
        )
        print(f"\n({msg})")


def print_bh_dataset_optimality_gap(
    *,
    global_max_yield: float,
    seed_best_yield: float,
    final_best_yield: float,
) -> None:
    """
    Compare best yield after BO to the dataset maximum for this discrete universe.

    For a Monte Carlo random baseline, see ``print_bh_optimality_and_random_baseline``.
    """
    print_banner("STEP 6 - OPTIMALITY CHECK (DATASET SCOPE)")
    print(f"Global max yield in universe (any row): {global_max_yield:.4f}")
    print(f"Best yield after initial training only: {seed_best_yield:.4f}")
    print(f"Best yield after BO campaign:           {final_best_yield:.4f}")
    gap_to_opt = global_max_yield - final_best_yield
    print(f"Gap to global optimum:                  {gap_to_opt:.4f}")
    if gap_to_opt <= 1e-6:
        print("Reached the dataset optimum within numerical tolerance.")
    elif gap_to_opt < 0.5:
        print("Very close to the dataset optimum for this run.")
    else:
        print(
            "Gap remains; try more N_BO_ITERATIONS, a different ACQUISITION rule, or adjust "
            "N_INITIAL_POINTS (or ``BH_POOL_BO_INIT_STRATEGY`` / ``BH_POOL_BO_MIN_NONZERO_LOW_POINTS`` in bo_common)."
        )


def summarize_bo_history_step_table(
    history_df: pd.DataFrame,
    target_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Per-step aggregates for trajectory plots (generic score column)."""
    step_summary = (
        history_df.groupby("step", as_index=False)
        .agg(
            best_before=("best_before", "first"),
            observed_mean=(target_col, "mean"),
            observed_std=(target_col, "std"),
            score_mean=(score_col, "mean"),
            score_std=(score_col, "std"),
            pred_std_mean=("pred_std", "mean"),
            pred_std_std=("pred_std", "std"),
            n_points=(score_col, "size"),
        )
        .sort_values("step")
        .reset_index(drop=True)
    )
    step_summary["observed_std"] = step_summary["observed_std"].fillna(0.0)
    step_summary["score_std"] = step_summary["score_std"].fillna(0.0)
    step_summary["pred_std_std"] = step_summary["pred_std_std"].fillna(0.0)
    return step_summary


def plot_bo_trajectory_figures(
    step_summary: pd.DataFrame,
    *,
    acquisition_label: str,
    score_ylabel: str,
) -> None:
    """Three figures: best vs observed batch, acquisition mean, predicted std mean."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        step_summary["step"],
        step_summary["best_before"],
        marker="o",
        label="Best observed before step",
    )
    plt.errorbar(
        step_summary["step"],
        step_summary["observed_mean"],
        yerr=step_summary["observed_std"],
        fmt="s-",
        capsize=4,
        label="Observed yield (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel("Yield")
    plt.title("Sequential BO trajectory (batch-aware)")
    plt.legend()
    _set_bo_step_axis_integer_ticks()
    show_current_figure()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        step_summary["step"],
        step_summary["score_mean"],
        yerr=step_summary["score_std"],
        fmt="o-",
        capsize=4,
        label=f"{acquisition_label} (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel(score_ylabel)
    plt.title("Acquisition value per BO step (batch-aware)")
    plt.legend()
    _set_bo_step_axis_integer_ticks()
    show_current_figure()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        step_summary["step"],
        step_summary["pred_std_mean"],
        yerr=step_summary["pred_std_std"],
        fmt="o-",
        capsize=4,
        label="Predicted std (mean +/- std within batch)",
    )
    plt.xlabel("BO step")
    plt.ylabel("Predicted uncertainty (std)")
    plt.title("Uncertainty per BO step (batch-aware)")
    plt.legend()
    _set_bo_step_axis_integer_ticks()
    show_current_figure()


def print_bh_optimality_and_random_baseline(
    *,
    global_max_yield: float,
    seed_best_yield: float,
    final_best_yield: float,
    random_median: float,
    random_mean: float,
    bo_budget: int,
    random_baseline_mc: int,
) -> None:
    """STEP 7B: dataset optimum vs BO vs Monte Carlo random baseline."""
    print_banner("STEP 7B - OPTIMALITY CHECK (DATASET SCOPE)")
    print(f"Global max yield in universe (any row): {global_max_yield:.4f}")
    print(f"Best yield after low-yield seed only:   {seed_best_yield:.4f}")
    print(f"Best yield after BO campaign:           {final_best_yield:.4f}")
    print(
        f"Random baseline ({bo_budget} picks, {random_baseline_mc} Monte Carlo runs): "
        f"median best={random_median:.4f}, mean best={random_mean:.4f}"
    )
    print(
        f"BO beat random (median of MC baselines): {final_best_yield >= random_median - 1e-9} "
        f"| BO beat random (mean): {final_best_yield >= random_mean - 1e-9}"
    )
    gap_to_opt = global_max_yield - final_best_yield
    print(f"Gap to global optimum:                  {gap_to_opt:.4f}")
    if gap_to_opt <= 1e-6:
        print("Reached the dataset optimum within numerical tolerance.")
    elif gap_to_opt < 0.5:
        print("Very close to the dataset optimum for this run.")
    else:
        print(
            "Gap remains; consider more BO steps, higher UCB_BETA, or ACQUISITION='ts'."
        )
