from __future__ import annotations

import gc
import os
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from typing import Optional

print("PYTHON EXECUTABLE:", sys.executable)
print("WORKING DIRECTORY:", os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as _Patch

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2

import argparse

RDLogger.DisableLog("rdApp.*")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Amide yield prediction — generational model comparison."
    )
    parser.add_argument(
        "--pb-path",
        type=str,
        default="/Users/jacobmoon/Documents/ord-data/data/47/ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz",
        help="Path to the ORD .pb.gz dataset file.",
    )
    return parser.parse_args()

PB_PATH = _parse_args().pb_path

SMILES_TYPE = 2
NAME_TYPE = 6
FP_BITS = 256
TEST_SIZE = 0.2
RANDOM_STATE = 42


def extract_yield(outcomes) -> Optional[float]:
    if outcomes is None:
        return None
    for outcome in outcomes:
        for product in outcome.products:
            for meas in product.measurements:
                try:
                    if meas.HasField("percentage"):
                        return float(meas.percentage.value)
                except Exception:
                    pass
    return None


def get_smiles_from_component(comp) -> Optional[str]:
    for ident in comp.identifiers:
        try:
            if ident.type == SMILES_TYPE and ident.value:
                return ident.value
        except Exception:
            continue
    return None


def get_name_from_component(comp) -> Optional[str]:
    for ident in comp.identifiers:
        try:
            if ident.type == NAME_TYPE and ident.value:
                return ident.value
        except Exception:
            continue
    return None


def preferred_label_for_component(comp) -> Optional[str]:
    name_val = get_name_from_component(comp)
    if name_val is not None:
        return name_val

    smiles_val = get_smiles_from_component(comp)
    if smiles_val is not None:
        return smiles_val

    for ident in comp.identifiers:
        try:
            if ident.value:
                return ident.value
        except Exception:
            continue
    return None


def component_has_moles(comp) -> bool:
    try:
        return comp.amount.HasField("moles")
    except Exception:
        return False


def component_has_volume(comp) -> bool:
    try:
        return comp.amount.HasField("volume")
    except Exception:
        return False


def extract_condition_label(inputs, target_key: str) -> str:
    if target_key not in inputs:
        return "MISSING"

    block = inputs[target_key]
    comps = list(block.components)

    if len(comps) == 0:
        return "MISSING"

    if target_key == "solvent":
        for comp in comps:
            if component_has_volume(comp):
                label = preferred_label_for_component(comp)
                if label:
                    return label
        label = preferred_label_for_component(comps[0])
        return label if label else "MISSING"

    for comp in comps:
        if component_has_moles(comp):
            label = preferred_label_for_component(comp)
            if label:
                return label

    for comp in comps:
        if not component_has_volume(comp):
            label = preferred_label_for_component(comp)
            if label:
                return label

    label = preferred_label_for_component(comps[-1])
    return label if label else "MISSING"


def extract_substrate_smiles(inputs, target_key: str) -> Optional[str]:
    if target_key not in inputs:
        return None

    block = inputs[target_key]
    comps = list(block.components)

    if len(comps) == 0:
        return None

    for comp in comps:
        if component_has_moles(comp):
            smi = get_smiles_from_component(comp)
            if smi:
                return smi

    for comp in comps:
        if not component_has_volume(comp):
            smi = get_smiles_from_component(comp)
            if smi:
                return smi

    for comp in comps:
        smi = get_smiles_from_component(comp)
        if smi:
            return smi

    return None


def mol_to_fp(smi: Optional[str], n_bits: int = FP_BITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    if smi is None:
        return arr

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return arr

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    out = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, out)
    return out.astype(np.float32)


def load_ord_amide_dataset(pb_path: str) -> pd.DataFrame:
    print("Reading protobuf dataset...")
    dataset = load_message(pb_path, dataset_pb2.Dataset)
    print("Finished reading protobuf dataset.")
    print("Total reactions in raw dataset:", len(dataset.reactions))

    rows = []
    for i, rxn in enumerate(dataset.reactions):
        if i % 5000 == 0:
            print(f"Processed {i} reactions...")

        amine_smi = extract_substrate_smiles(rxn.inputs, "amine")
        acid_smi = extract_substrate_smiles(rxn.inputs, "carboxylic acid")
        rxn_yield = extract_yield(rxn.outcomes)

        if amine_smi is None or acid_smi is None or rxn_yield is None:
            continue

        rows.append(
            {
                "reaction_id": rxn.reaction_id,
                "amine_smi": amine_smi,
                "acid_smi": acid_smi,
                "yield": rxn_yield,
                "solvent": extract_condition_label(rxn.inputs, "solvent"),
                "base": extract_condition_label(rxn.inputs, "base"),
                "additive": extract_condition_label(rxn.inputs, "additive"),
                "activation_agent": extract_condition_label(rxn.inputs, "activation agent"),
            }
        )

    print("Finished extracting usable reactions.")
    return pd.DataFrame(rows)


def make_condition_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    condition_cols = ["solvent", "base", "additive", "activation_agent"]
    varying_cols = [col for col in condition_cols if df[col].nunique() > 1]
    print("Condition columns used:", varying_cols)

    cond_df = pd.get_dummies(
        df[varying_cols],
        columns=varying_cols,
        prefix=varying_cols,
        dtype=int,
    )
    return cond_df, varying_cols


def make_amine_features(df: pd.DataFrame) -> np.ndarray:
    fps = []
    for i, s in enumerate(df["amine_smi"]):
        if i % 10000 == 0:
            print(f"Amine fingerprints: {i} / {len(df)}")
        fps.append(mol_to_fp(s))
    return np.stack(fps, axis=0)


def make_acid_features(df: pd.DataFrame) -> np.ndarray:
    fps = []
    for i, s in enumerate(df["acid_smi"]):
        if i % 10000 == 0:
            print(f"Acid fingerprints: {i} / {len(df)}")
        fps.append(mol_to_fp(s))
    return np.stack(fps, axis=0)


# ---------------------------------------------------------------------------
# Gen 8: sklearn MLP (no PyTorch dependency, works on all platforms)
# ---------------------------------------------------------------------------

class MLPRegressorWrapper:
    """
    Thin wrapper around sklearn MLPRegressor that adds:
    - StandardScaler (fingerprint features benefit from normalization)
    - feature_importances_ stub (returns zeros; MLP has no native importance)
    - per-epoch progress printing via the warm_start trick
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,          # L2 regularization
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=1,          # we iterate manually so we can print progress
            warm_start=True,     # keeps weights between partial fit calls
            random_state=random_state,
            verbose=False,
        )
        self._n_features: int = 0

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(self._n_features, dtype=np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPRegressorWrapper":
        self._n_features = x.shape[1]
        print(f"    MLP: scaling features {x.shape}...", flush=True)
        x_scaled = self.scaler.fit_transform(x)

        n_epochs = 50
        print(f"    MLP: training for {n_epochs} epochs...", flush=True)
        for epoch in range(1, n_epochs + 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.model_.max_iter = epoch
                self.model_.fit(x_scaled, y)
            if epoch % 10 == 0:
                train_preds = self.model_.predict(x_scaled)
                mse = np.mean((train_preds - y) ** 2)
                print(f"    MLP epoch {epoch}/{n_epochs}  MSE={mse:.2f}", flush=True)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        return self.model_.predict(x_scaled)



def evaluate_model(model_name: str, x: np.ndarray, y: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    if model_name == "linear_regression":
        model = LinearRegression()

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "xgboost_basic":
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    elif model_name == "xgboost_tuned":
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    elif model_name == "mlp":
        model = MLPRegressorWrapper(random_state=RANDOM_STATE)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return mae, r2, model, x_train, x_test, y_train, y_test, preds


def make_feature_names(cond_df: pd.DataFrame) -> list[str]:
    amine_names = [f"amine_fp_{i}" for i in range(FP_BITS)]
    acid_names = [f"acid_fp_{i}" for i in range(FP_BITS)]
    cond_names = cond_df.columns.tolist()
    return amine_names + acid_names + cond_names


# ---------------------------------------------------------------------------
# FP size sweep
# ---------------------------------------------------------------------------

FP_SWEEP_SIZES = [64, 128, 256, 512, 1024, 2048]


def _build_full_features_at_bits(
    df: pd.DataFrame,
    x_cond: np.ndarray,
    n_bits: int,
) -> np.ndarray:
    """Rebuild amine + acid fingerprints at a given bit size and concatenate conditions."""
    amine_fps, acid_fps = [], []
    for smi in df["amine_smi"]:
        amine_fps.append(mol_to_fp(smi, n_bits=n_bits))
    for smi in df["acid_smi"]:
        acid_fps.append(mol_to_fp(smi, n_bits=n_bits))
    x_amine = np.stack(amine_fps, axis=0)
    x_acid = np.stack(acid_fps, axis=0)
    return np.concatenate([x_amine, x_acid, x_cond], axis=1)


def run_fp_size_sweep(
    df: pd.DataFrame,
    x_cond: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """
    Train the tuned XGBoost (Gen 7 config) at each FP bit size in FP_SWEEP_SIZES.
    Returns a DataFrame with columns: fp_bits, MAE, R2.
    """
    print("\n=== FP Size Sweep ===")
    sweep_rows = []

    for n_bits in FP_SWEEP_SIZES:
        print(f"  Building features at {n_bits} bits...")
        x = _build_full_features_at_bits(df, x_cond, n_bits)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"    fp_bits={n_bits:5d}  R²={r2:.4f}  MAE={mae:.4f}")
        sweep_rows.append({"fp_bits": n_bits, "MAE": mae, "R2": r2})

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("fp_sweep_results.csv", index=False)
    print("Saved fp_sweep_results.csv")
    return sweep_df


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------

# Color palette — consistent across all figures
_C_CONDITIONS = "#3b82f6"   # blue  — conditions only
_C_STRUCTURE  = "#f97316"   # orange — structure only
_C_COMBINED   = "#22c55e"   # green  — combined
_C_DL         = "#a855f7"   # purple — deep learning

_GEN_COLORS = [
    _C_CONDITIONS,  # Gen 1 — conditions, linear regression
    _C_CONDITIONS,  # Gen 2 — conditions, random forest
    _C_CONDITIONS,  # Gen 3 — conditions, XGBoost
    _C_STRUCTURE,   # Gen 4 — amine FP only
    _C_STRUCTURE,   # Gen 5 — acid FP only
    _C_STRUCTURE,   # Gen 6 — both FPs
    _C_COMBINED,    # Gen 7 — full features, XGBoost basic
    _C_COMBINED,    # Gen 8 — full features, tuned XGBoost
    _C_DL,          # Gen 9 — MLP
]

_LEGEND_ELEMENTS = [
    _Patch(facecolor=_C_CONDITIONS, label="Conditions only"),
    _Patch(facecolor=_C_STRUCTURE,  label="Structure only"),
    _Patch(facecolor=_C_COMBINED,   label="Combined"),
    _Patch(facecolor=_C_DL,         label="Deep learning (MLP)"),
]


def plot_figure2_ablation(results_in_order: list[dict]) -> None:
    """
    Figure 2: Controlled XGBoost ablation — all five generations that use
    xgboost_basic, isolating the feature set contribution with model class held fixed.
    """
    keys = ["Gen 3:", "Gen 4:", "Gen 5:", "Gen 6:", "Gen 7:"]
    short_labels = [
        "Conditions\nonly",
        "Amine FP\nonly",
        "Acid FP\nonly",
        "Both FPs\nno conditions",
        "Both FPs\n+ conditions",
    ]
    colors = [
        _C_CONDITIONS,
        _C_STRUCTURE,
        _C_STRUCTURE,
        _C_STRUCTURE,
        _C_COMBINED,
    ]

    rows = [r for r in results_in_order if any(r["generation"].startswith(k) for k in keys)]
    r2_vals = [r["R2"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(short_labels, r2_vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)

    ax.annotate(
        f"+{r2_vals[4] - r2_vals[3]:.2f} R²\nwhen conditions added",
        xy=(4, r2_vals[4]),
        xytext=(3.35, r2_vals[4] - 0.13),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, color="black",
    )

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"R²={val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("R² (Test Set)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Figure 2: Controlled Ablation — XGBoost, Feature Set Varied\n"
        "Same model class across all bars; only the input features change",
        fontsize=12, pad=12,
    )
    legend_elements = [
        _Patch(facecolor=_C_CONDITIONS, label="Conditions only"),
        _Patch(facecolor=_C_STRUCTURE,  label="Structure only"),
        _Patch(facecolor=_C_COMBINED,   label="Combined"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    fig.tight_layout()
    plt.savefig("figure2_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure2_ablation.png")


def plot_figure1_progression(results_in_order: list[dict]) -> None:
    """
    Figure 1: R² line chart across all generations in order.
    """
    gen_nums = list(range(1, len(results_in_order) + 1))
    r2_vals  = [r["R2"] for r in results_in_order]
    colors   = _GEN_COLORS

    gen_labels = [f"Gen {i}" for i in gen_nums]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvspan(0.5, 3.5, alpha=0.07, color=_C_CONDITIONS, label="_nolegend_")
    ax.axvspan(3.5, 6.5, alpha=0.07, color=_C_STRUCTURE,  label="_nolegend_")
    ax.axvspan(6.5, 8.5, alpha=0.07, color=_C_COMBINED,   label="_nolegend_")
    ax.axvspan(8.5, 9.5, alpha=0.07, color=_C_DL,         label="_nolegend_")

    ax.plot(gen_nums, r2_vals, color="black", linewidth=1.5,
            zorder=2, linestyle="--", alpha=0.4)

    for gn, r2, col in zip(gen_nums, r2_vals, colors):
        ax.scatter(gn, r2, color=col, s=120, zorder=3, edgecolors="white", linewidths=1.2)
        ax.text(gn, r2 + 0.025, f"{r2:.3f}", ha="center", fontsize=8.5)

    ax.set_xticks(gen_nums)
    ax.set_xticklabels(gen_labels, fontsize=9)
    ax.set_ylabel("R² (Test Set)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, 9.5)
    ax.set_title(
        "Figure 1: R² Across All Model Generations",
        fontsize=12, pad=12,
    )
    ax.legend(handles=_LEGEND_ELEMENTS, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.savefig("figure1_progression.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure1_progression.png")


def plot_figure4_scatter_comparison(
    gen7_payload: dict,
    best_payload: dict,
) -> None:
    """
    Figure 4: Side-by-side predicted vs actual for Gen 8 XGBoost and Gen 9 MLP.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for ax, payload, label, color in [
        (ax1, gen7_payload,  "Gen 8: Tuned XGBoost", _C_COMBINED),
        (ax2, best_payload,  "Gen 9: MLP",           _C_DL),
    ]:
        hb = ax.hexbin(
            payload["y_test"], payload["preds"],
            gridsize=35, cmap="Blues", mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="Count")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Actual Yield (%)", fontsize=11)
        ax.set_ylabel("Predicted Yield (%)", fontsize=11)
        ax.set_title(
            f"{label}\nR²={payload['r2']:.3f}   MAE={payload['mae']:.2f}%",
            fontsize=11,
        )

    fig.suptitle(
        "Figure 4: Predicted vs. Actual Yield — XGBoost vs. MLP",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    plt.savefig("figure4_scatter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure4_scatter_comparison.png")


def plot_figure3_feature_group_importance(
    gen7_model,
    feature_names: list[str],
) -> None:
    """
    Figure 3: Feature importance grouped into amine FP, acid FP, and conditions.
    """
    importances = gen7_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    amine_total = fi_df[fi_df["feature"].str.startswith("amine_fp")]["importance"].sum()
    acid_total  = fi_df[fi_df["feature"].str.startswith("acid_fp")]["importance"].sum()
    cond_total  = fi_df[
        ~fi_df["feature"].str.startswith("amine_fp") &
        ~fi_df["feature"].str.startswith("acid_fp")
    ]["importance"].sum()
    total = amine_total + acid_total + cond_total

    groups = ["Amine\nFingerprint", "Acid\nFingerprint", "Reaction\nConditions"]
    vals   = [amine_total / total, acid_total / total, cond_total / total]
    colors = [_C_CONDITIONS, _C_STRUCTURE, _C_COMBINED]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(groups, vals, color=colors, width=0.45,
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.1%}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Share of Total Feature Importance", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Figure 3: Feature Group Importance — Gen 8 Tuned XGBoost",
        fontsize=12, pad=12,
    )
    fig.tight_layout()
    plt.savefig("figure3_feature_group_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure3_feature_group_importance.png")


def plot_figure5_fp_sweep(sweep_df: pd.DataFrame) -> None:
    """
    Figure 5: FP size sensitivity — R² and MAE vs bit width (log scale).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Morgan Fingerprint Size (bits, log scale)", fontsize=11)
    ax1.set_ylabel("R²", color=_C_COMBINED, fontsize=11)
    ax1.semilogx(sweep_df["fp_bits"], sweep_df["R2"],
                 marker="o", color=_C_COMBINED, linewidth=2, label="R²")
    ax1.tick_params(axis="y", labelcolor=_C_COMBINED)
    ax1.set_xticks(sweep_df["fp_bits"])
    ax1.set_xticklabels(sweep_df["fp_bits"])

    ax2 = ax1.twinx()
    ax2.set_ylabel("MAE (yield %)", color=_C_STRUCTURE, fontsize=11)
    ax2.semilogx(sweep_df["fp_bits"], sweep_df["MAE"],
                 marker="s", linestyle="--", color=_C_STRUCTURE,
                 linewidth=2, label="MAE")
    ax2.tick_params(axis="y", labelcolor=_C_STRUCTURE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title(
        "Figure 5: Fingerprint Size Sensitivity (Gen 8 Tuned XGBoost)",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    plt.savefig("figure5_fp_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure5_fp_sweep.png")


def main():
    print(f"Loading dataset from: {PB_PATH}")

    df = load_ord_amide_dataset(PB_PATH)

    if df.empty:
        raise ValueError("No usable reactions found in dataset.")

    print(f"Number of usable reactions: {len(df)}")
    print(df[[
        "reaction_id",
        "amine_smi",
        "acid_smi",
        "solvent",
        "base",
        "additive",
        "activation_agent",
        "yield",
    ]].head())

    print("\nUnique values per condition column:")
    for col in ["solvent", "base", "additive", "activation_agent"]:
        print(f"{col}: {df[col].nunique()} unique values")

    df.to_csv("amide_reactions_readable_full.csv", index=False)
    print("Saved parsed dataset to amide_reactions_readable_full.csv")

    y = df["yield"].astype(float).values

    print("\nBuilding condition features...")
    cond_df, varying_cols = make_condition_features(df)
    x_cond = cond_df.to_numpy(dtype=np.float32)

    print("\nBuilding amine fingerprints...")
    x_amine = make_amine_features(df)

    print("\nBuilding acid fingerprints...")
    x_acid = make_acid_features(df)

    print("\nCombining feature matrices...")
    x_both_fp = np.concatenate([x_amine, x_acid], axis=1)
    x_full = np.concatenate([x_amine, x_acid, x_cond], axis=1)

    print("\nFeature summary:")
    print("Condition feature shape:", x_cond.shape)
    print("Amine feature shape:", x_amine.shape)
    print("Acid feature shape:", x_acid.shape)
    print("Both substrate feature shape:", x_both_fp.shape)
    print("Full feature shape:", x_full.shape)

    experiments = [
        (
            "Gen 1: conditions only + linear regression",
            x_cond,
            "linear_regression",
            "Only varying reaction conditions",
        ),
        (
            "Gen 2: conditions only + random forest",
            x_cond,
            "random_forest",
            "Nonlinear model on varying reaction conditions",
        ),
        (
            "Gen 3: conditions only + XGBoost",
            x_cond,
            "xgboost_basic",
            "Conditions only — same model class as Gens 4-7 for controlled comparison",
        ),
        (
            "Gen 4: amine fp only + XGBoost",
            x_amine,
            "xgboost_basic",
            "Structure of amine only",
        ),
        (
            "Gen 5: acid fp only + XGBoost",
            x_acid,
            "xgboost_basic",
            "Structure of acid only",
        ),
        (
            "Gen 6: amine + acid fp + XGBoost",
            x_both_fp,
            "xgboost_basic",
            "Both substrates, no conditions",
        ),
        (
            "Gen 7: full features + XGBoost",
            x_full,
            "xgboost_basic",
            "Substrates + varying conditions",
        ),
        (
            "Gen 8: full features + tuned XGBoost",
            x_full,
            "xgboost_tuned",
            "Substrates + varying conditions + tuned hyperparameters",
        ),
        (
            "Gen 9: full features + MLP (deep learning)",
            x_full,
            "mlp",
            "3-layer MLP with adaptive learning rate and L2 regularization",
        ),
    ]

    results = []
    best_model = None
    best_r2 = -np.inf
    best_payload = None
    gen7_payload = None    # predictions + model for Gen 7 XGBoost
    results_in_order = []  # preserves generation order for line chart

    for generation, x, model_name, change in experiments:
        print(f"\nRunning {generation} ...", flush=True)
        mae, r2, model, x_train, x_test, y_train, y_test, preds = evaluate_model(model_name, x, y)

        row = {
            "generation": generation,
            "change_made": change,
            "n_features": x.shape[1],
            "model": model_name,
            "MAE": mae,
            "R2": r2,
        }
        results.append(row)
        results_in_order.append(row)

        print(f"  Change: {change}")
        print(f"  Features: {x.shape}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_payload = {
                "generation": generation,
                "mae": mae,
                "r2": r2,
                "y_test": y_test,
                "preds": preds,
                "x_shape": x.shape,
            }

        # save Gen 8 tuned XGBoost separately for feature importance + scatter
        if "Gen 8" in generation and "tuned" in generation:
            gen7_payload = {
                "model": model,
                "y_test": y_test,
                "preds": preds,
                "r2": r2,
                "mae": mae,
            }

        # free memory between generations
        del model
        gc.collect()

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    print("\n=== Final Comparison Table ===")
    print(results_df)

    results_df.to_csv("model_generation_results_full.csv", index=False)
    print("\nSaved results to model_generation_results_full.csv")

    print(f"\nBest model: {best_payload['generation']}")
    print(f"Best model R2: {best_payload['r2']:.4f}")
    print(f"Best model MAE: {best_payload['mae']:.4f}")

    feature_names = make_feature_names(cond_df)

    # --- Five publication figures ---
    plot_figure1_progression(results_in_order)
    plot_figure2_ablation(results_in_order)

    if gen7_payload is not None:
        plot_figure3_feature_group_importance(gen7_payload["model"], feature_names)
        plot_figure4_scatter_comparison(gen7_payload, best_payload)
    else:
        print("Gen 8 payload not captured — skipping figures 3 and 4.")

    # ------------------------------------------------------------------
    # FP size sensitivity analysis
    # ------------------------------------------------------------------
    sweep_df = run_fp_size_sweep(df, x_cond, y)
    plot_figure5_fp_sweep(sweep_df)

    print("\n=== FP Sweep Results ===")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()