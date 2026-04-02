from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2

RDLogger.DisableLog("rdApp.*")

SMILES_TYPE = 2
NAME_TYPE = 6


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


def mol_to_fp(smi: Optional[str], n_bits: int) -> np.ndarray:
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


def load_ord_amide_dataset(pb_path: Path) -> pd.DataFrame:
    print(f"Reading protobuf dataset from: {pb_path}")
    dataset = load_message(str(pb_path), dataset_pb2.Dataset)
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


def make_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    condition_cols = ["solvent", "base", "additive", "activation_agent"]
    varying_cols = [col for col in condition_cols if df[col].nunique() > 1]
    print("Condition columns used:", varying_cols)

    return pd.get_dummies(
        df[varying_cols],
        columns=varying_cols,
        prefix=varying_cols,
        dtype=int,
    )


def make_amine_features(df: pd.DataFrame, fp_bits: int) -> np.ndarray:
    fps = []
    for i, s in enumerate(df["amine_smi"]):
        if i % 10000 == 0:
            print(f"Amine fingerprints: {i} / {len(df)}")
        fps.append(mol_to_fp(s, fp_bits))
    return np.stack(fps, axis=0)


def make_acid_features(df: pd.DataFrame, fp_bits: int) -> np.ndarray:
    fps = []
    for i, s in enumerate(df["acid_smi"]):
        if i % 10000 == 0:
            print(f"Acid fingerprints: {i} / {len(df)}")
        fps.append(mol_to_fp(s, fp_bits))
    return np.stack(fps, axis=0)


def make_feature_names(cond_df: pd.DataFrame, fp_bits: int) -> list[str]:
    amine_names = [f"amine_fp_{i}" for i in range(fp_bits)]
    acid_names = [f"acid_fp_{i}" for i in range(fp_bits)]
    cond_names = cond_df.columns.tolist()
    return amine_names + acid_names + cond_names


def train_gen7_model(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
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
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, y_test, preds, mae, r2


def plot_predicted_vs_actual(
    y_test: np.ndarray,
    preds: np.ndarray,
    r2: float,
    mae: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.25, s=12)
    plt.plot([0, 100], [0, 100], linestyle="--")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Generation 7: Predicted vs Actual\nR² = {r2:.3f}, MAE = {mae:.2f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def save_feature_importance(
    model,
    feature_names: list[str],
    csv_path: Path,
    png_path: Path,
    top_n: int = 30,
) -> None:
    importances = model.feature_importances_

    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fi_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    top_df = fi_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (Generation 7)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Generation 7 amide-yield model on an ORD dataset."
    )
    parser.add_argument(
        "--pb-path",
        type=Path,
        required=True,
        help="Path to the ORD .pb.gz dataset file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint size.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for the test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_ord_amide_dataset(args.pb_path)

    if df.empty:
        raise ValueError("No usable reactions found in dataset.")

    print(f"Number of usable reactions: {len(df)}")
    print(
        df[
            [
                "reaction_id",
                "amine_smi",
                "acid_smi",
                "solvent",
                "base",
                "additive",
                "activation_agent",
                "yield",
            ]
        ].head()
    )

    print("\nUnique values per condition column:")
    for col in ["solvent", "base", "additive", "activation_agent"]:
        print(f"{col}: {df[col].nunique()} unique values")

    readable_csv = results_dir / "gen7_readable_dataset.csv"
    df.to_csv(readable_csv, index=False)
    print(f"Saved {readable_csv}")

    y = df["yield"].astype(float).values

    print("\nBuilding condition features...")
    cond_df = make_condition_features(df)
    x_cond = cond_df.to_numpy(dtype=np.float32)

    print("\nBuilding amine fingerprints...")
    x_amine = make_amine_features(df, args.fp_bits)

    print("\nBuilding acid fingerprints...")
    x_acid = make_acid_features(df, args.fp_bits)

    print("\nCombining full feature matrix...")
    x_full = np.concatenate([x_amine, x_acid, x_cond], axis=1)

    print("\nFeature summary:")
    print("Condition feature shape:", x_cond.shape)
    print("Amine feature shape:", x_amine.shape)
    print("Acid feature shape:", x_acid.shape)
    print("Full feature shape:", x_full.shape)

    print("\nTraining Generation 7 model...")
    model, y_test, preds, mae, r2 = train_gen7_model(
        x_full,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n=== Generation 7 Results ===")
    print("Model: tuned XGBoost with full features")
    print(f"FP_BITS: {args.fp_bits}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    plot_predicted_vs_actual(
        y_test,
        preds,
        r2,
        mae,
        results_dir / "gen7_predicted_vs_actual.png",
    )

    feature_names = make_feature_names(cond_df, args.fp_bits)
    save_feature_importance(
        model,
        feature_names,
        results_dir / "gen7_feature_importance.csv",
        results_dir / "gen7_top30_feature_importance.png",
        top_n=30,
    )


if __name__ == "__main__":
    main()