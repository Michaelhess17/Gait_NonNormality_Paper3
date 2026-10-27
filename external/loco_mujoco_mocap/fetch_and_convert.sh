#!/usr/bin/env bash
# Download the LocoMuJoCo mocap files used by Figure 4 and convert them to the
# joint-angle CSVs in processed/. The dataset is CC-BY-NC-ND-4.0, so it is not
# redistributed here; this fetches the unmodified originals from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"
base="https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/resolve/main/DefaultDatasets/mocap"
for f in Atlas/walk Atlas/run UnitreeH1/walk UnitreeH1/run UnitreeH1/random_walk UnitreeH1/stepinplace1 UnitreeG1/walk; do
  mkdir -p "raw/$(dirname "$f")"
  [ -s "raw/$f.npz" ] || curl -fL --retry 3 -o "raw/$f.npz" "$base/$f.npz"
done
python3 convert_to_csv.py
