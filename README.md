Generation 7 Amide Yield Model

This project trains a tuned XGBoost model to predict amide coupling yield from:

amine Morgan fingerprints
acid Morgan fingerprints
one-hot encoded reaction-condition features
Tested Environment

Tested with Python 3.9+.

Files in This Repository
gen7_model.py — main script
requirements.txt — Python dependencies
README.md — setup and run instructions

1. Download This Repository

Either download the ZIP from GitHub and unzip it, or clone it with:

git clone https://github.com/jwm2181/Maddie-Jacob-Applied-ML-Final.git
cd Maddie-Jacob-Applied-ML-Final

2. Install Python Dependencies

Create a Python environment if desired, then run:

pip install -r requirements.txt

If rdkit fails to install with pip, using a Conda environment is often more reliable.

3. Install Git LFS

The ORD data repository uses Git LFS.

On Mac
brew install git-lfs
git lfs install

4. Download the ORD Dataset

Clone the ORD data repository:

git clone https://github.com/open-reaction-database/ord-data.git

Find the dataset file:

find ord-data -name "ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz"

Create a local data/ folder inside this project and copy the dataset into it:

mkdir -p data
cp ord-data/data/47/ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz data/

5. Run the Model
python gen7_model.py --pb-path data/ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41.pb.gz

6. Outputs

The script writes results to the results/ folder:

gen7_readable_dataset.csv
gen7_predicted_vs_actual.png
gen7_feature_importance.csv
gen7_top30_feature_importance.png
Notes
The full ORD dataset is not included in this repository.
It must be downloaded separately using the instructions above.
The script expects the dataset in .pb.gz format.
