"""
Pool-based Bayesian optimization on the BH HTE dataset.

- Run with **cwd = this session folder** (``Session_7_Bayesian_Opt``) so the CSV path in User settings resolves.
- Keep this script next to ``bo_common.py`` (import bootstrap uses ``Path(__file__).parent``).
- Search space is the unique rows of the lesson CSV (no synthetic grid).
- Initial train/pool split, acquisition, batch size, iterations, and random seed are set in the **User settings** block.
- Each BO step refits ROBERT and ranks the remaining pool by the configured acquisition rule.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

_SESSION_DIR = Path(__file__).resolve().parent
if not (_SESSION_DIR / "bo_common.py").is_file():
    raise ImportError(
        "bo_common.py must sit beside this script. Place BH.py in Session_7_Bayesian_Opt next to bo_common.py."
    )
if str(_SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SESSION_DIR))

import bo_common  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")

# ------------------------------- User settings ------------------------------- #
# Basename only: run the script with cwd = Session_7_Bayesian_Opt.
CSV_FILE = "BH_HTE_small.csv"
TARGET_COL = "Yield"
ID_COLS = ("Reaction",)
# Explicit model inputs (must match ``BH_HTE_small.csv`` headers; ``Reaction`` is only an ID).
FEATURE_COLS = (
    "Substrate",
    "Substrate_MolWt",
    "Substrate_MolLogP",
    "Substrate_TPSA",
    "Catalyst",
    "Base",
    "Solvent",
    "Solvent_MolWt",
    "Solvent_MolLogP",
    "Solvent_TPSA",
)
CAT_COLS = ("Substrate", "Catalyst", "Base", "Solvent")
NUM_COLS = (
    "Substrate_MolWt",
    "Substrate_MolLogP",
    "Substrate_TPSA",
    "Solvent_MolWt",
    "Solvent_MolLogP",
    "Solvent_TPSA",
)
# Merged into ``RobertModel(...)`` after ``bo_common.ROBERT_BO_FAST_KW`` (see ROBERT docs).
# Optional ``workdir``: fixed path for ROBERT outputs; default is a new folder under ``./robert_bo_outputs/``.
ROBERT_MODEL_KW: dict[str, Any] = {}
FULL_CSV_INSPECTION = True

# Reproducibility and BO loop length (integer seed for numpy/sklearn stochastic pieces)
RANDOM_SEED = 7
N_BO_ITERATIONS = 5
BATCH_SIZE = 5

# Acquisition rule for ranking the pool: "ucb" | "ei" | "ts" | "random"
ACQUISITION = "ts"

# Initial training set: "worst" (pedagogical low-yield seed) or "random" (uniform sample from universe).
INITIAL_TRAIN_STRATEGY: Literal["worst", "random"] = "worst"
N_INITIAL_POINTS = 10
# Worst-seed constraint uses ``bo_common.BH_POOL_BO_MIN_NONZERO_LOW_POINTS``.
# UCB / EI shaping follows ``bo_common.BH_POOL_BO_UCB_BETA`` / ``BH_POOL_BO_EI_XI``.


def main() -> None:
    rs = RANDOM_SEED
    n_bo_steps = N_BO_ITERATIONS
    batch_size = BATCH_SIZE
    acquisition_mode = ACQUISITION
    acquisition_label = bo_common.bh_pool_bo_acquisition_label(acquisition_mode)

    bo_common.print_banner("STEP 1 - LOAD DATA")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded CSV from: {CSV_FILE}")
    bo_common.quick_inspect_csv(df, target_col=TARGET_COL)
    if FULL_CSV_INSPECTION:
        bo_common.print_full_dataframe_inspection(df)

    bo_common.print_banner("STEP 2 - DEFINE FEATURE COLUMNS")
    feature_cols = list(FEATURE_COLS)
    cat_cols = list(CAT_COLS)
    num_cols = list(NUM_COLS)
    bo_common.validate_bo_feature_columns(df, TARGET_COL, feature_cols, cat_cols, num_cols)
    print(f"Target column: {TARGET_COL}")
    print(f"Excluded ID columns (not used as features): {ID_COLS}")
    print(f"Feature columns: {feature_cols}")
    print(f"Categorical columns: {cat_cols}")
    print(f"Numeric columns: {num_cols}")

    bo_common.print_banner("STEP 3 - UNIVERSE AND INITIAL TRAIN/POOL SPLIT")
    universe_df = bo_common.build_unique_feature_universe(
        df, feature_cols, TARGET_COL, head_rows=10, verbose=True
    )
    observed_df, candidate_df = bo_common.build_bo_initial_observed_and_pool(
        universe_df,
        feature_cols,
        TARGET_COL,
        strategy=INITIAL_TRAIN_STRATEGY,
        n_points=N_INITIAL_POINTS,
        random_state=rs,
    )
    print("\nObserved training rows (head):")
    print(observed_df.head())
    print("\nCandidate rows (head):")
    print(candidate_df.head())

    bo_common.print_banner("STEP 4 - SURROGATE MODEL TRAINING")
    print(
        "Each BO iteration runs CURATE→GENERATE→VERIFY→PREDICT on the current training table using ROBERT."
    )

    global_max_yield = float(universe_df[TARGET_COL].max())
    seed_best_yield = float(observed_df[TARGET_COL].max())

    bo_common.print_banner("STEP 5 - SEQUENTIAL BO")
    bo_train = observed_df.copy()
    bo_pool = candidate_df.copy()
    selected_history: list[dict] = []
    score_col = "acq"
    print(
        f"ROBERT | acquisition: {acquisition_mode!r} ({acquisition_label}) "
        f"| steps={n_bo_steps} batch={batch_size}"
    )
    print(
        "Each iteration: rank pool → select batch → plot → append to training and shrink pool."
    )

    bo_common.print_banner("STEP 5A - ITERATION HELPERS")

    build_ranking_table = bo_common.make_robert_pool_ranking_fn(
        feature_cols,
        TARGET_COL,
        cat_cols,
        num_cols,
        random_state=rs,
        acquisition_mode=acquisition_mode,
        score_col=score_col,
        ucb_beta=bo_common.BH_POOL_BO_UCB_BETA,
        ei_xi=bo_common.BH_POOL_BO_EI_XI,
        robert_model_kwargs=ROBERT_MODEL_KW,
    )

    reveal_yield = bo_common.make_reveal_yield_from_dataset(
        universe_df, feature_cols, TARGET_COL
    )
    choose_batch_and_record = bo_common.make_choose_batch_from_ranking_fn(
        feature_cols,
        TARGET_COL,
        score_col,
        reveal_yield,
        "dataset_revealed",
    )

    bo_common.print_banner("STEP 5B - BO LOOP")
    # Rank pool → select batch → plot diagnostics → append to training and shrink pool (repeat).
    for step in range(1, n_bo_steps + 1):
        if len(bo_pool) == 0:
            print("Pool is empty; stopping early.")
            break

        bo_common.print_banner(f"BO ITERATION {step}")
        ranking, best_so_far = build_ranking_table(bo_train, bo_pool)
        current_batch_size = min(batch_size, len(ranking))
        print(f"Current best observed yield: {best_so_far:.3f}")
        print(f"Batch size this iteration: {current_batch_size}")
        print(f"Top 5 candidates by {acquisition_label}:")
        print(ranking.head(5))

        chosen_records, chosen_keys, history_items, selected_batch = (
            choose_batch_and_record(ranking, step, current_batch_size)
        )
        for history_item in history_items:
            history_item["best_before"] = best_so_far

        bo_common.plot_iteration_diagnostics(
            ranking=ranking,
            selected_batch=selected_batch,
            feature_cols=feature_cols,
            step=step,
            best_so_far=best_so_far,
            batch_size=current_batch_size,
            score_col=score_col,
            acquisition_label=acquisition_label,
        )
        bo_common.print_iteration_batch_summary(
            ranking, history_items, TARGET_COL, current_batch_size
        )

        selected_history.extend(history_items)
        bo_train, bo_pool = bo_common.update_train_and_pool(
            bo_train, bo_pool, chosen_records, chosen_keys, feature_cols
        )
        print(f"Training rows: {len(bo_train)} | Pool rows: {len(bo_pool)}")

    bo_common.print_pool_bo_campaign_recap(acquisition_label, batch_size, mode="dataset")

    history_df = pd.DataFrame(selected_history)
    final_best_yield = float(bo_train[TARGET_COL].max())

    bo_common.print_bh_dataset_optimality_gap(
        global_max_yield=global_max_yield,
        seed_best_yield=seed_best_yield,
        final_best_yield=final_best_yield,
    )

    bo_common.print_bo_history_tables_and_plots(
        history_df,
        TARGET_COL,
        score_col,
        acquisition_label,
        empty_iters_hint=(
            "No BO iterations ran; set a positive N_BO_ITERATIONS in User settings to populate history."
        ),
    )

    bo_common.print_banner("DONE")
    print("Script completed successfully.")
    print(
        "Tip: this script simulates sequential experimentation by hiding true yields until queried."
    )


if __name__ == "__main__":
    main()
