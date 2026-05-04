# Amide Coupling Yield Prediction — Generational Model Study

This project systematically investigates what drives amide coupling yield 
prediction through a 9-generation ablation study, varying feature sets, 
model classes, and hyperparameters.

## Tested Environment
Python 3.9+

## Files in This Repository
amide_yield_model2.py — main script (all 9 generations + figures)
requirements.txt — Python dependencies
README.md — setup and run instructions

## 1. Clone This Repository
git clone https://github.com/jwm2181/Maddie-Jacob-Applied-ML-Final.git
cd Maddie-Jacob-Applied-ML-Final

## 2. Install Dependencies
pip install -r requirements.txt

If rdkit fails with pip, use Conda:
conda install -c conda-forge rdkit

## 3. Install Git LFS
brew install git-lfs
git lfs install

## 4. Download the ORD Dataset
git clone https://github.com/open-reaction-database/ord-data.git
mkdir -p data
cp ord-data/data/47/ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz data/

## 5. Run
python amide_yield_model2.py \
  --pb-path data/ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz

## 6. Outputs
figure1_progression.png — R² across all 9 generations
figure2_ablation.png — controlled XGBoost feature ablation
figure3_feature_group_importance.png — feature group importance
figure4_scatter_comparison.png — XGBoost vs MLP predicted vs actual
figure5_fp_sweep.png — fingerprint size sensitivity
model_generation_results_full.csv — all generation metrics
fp_sweep_results.csv — fingerprint sweep metrics

## Notes
The ORD dataset is not included. Download separately using the 
instructions above. Dataset must be in .pb.gz format.
