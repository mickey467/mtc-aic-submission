# SGLATrack Competition Submission

Official submission for the MTC-AIC 2026 competition using the **SGLATrack** (Similarity-Guided Layer-Adaptive Vision Transformer) architecture.

## 🚀 Model Checkpoint Download
**Download Link:** [(https://drive.google.com/drive/folders/1FITQH29Jp3Xik9Ta9gwS-PaMb0Hc-aqp?usp=sharing)]  
Please download the `sglatrack_ep0297.pth.tar` file and place it in the `checkpoints/` directory.

## 🛠️ Installation
Ensure you have Python 3.8+ and a CUDA-enabled environment. Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🏃 Inference
To generate a submission file from the competition manifest, run the following command:
```bash
python inference.py \
    --weights checkpoints/sglatrack_ep0297.pth.tar \
    --data_root /path/to/dataset/root \
    --manifest /path/to/contestant_manifest.json \
    --split public_lb \
    --output submission.csv
```
*Note: The script includes compatibility patches for PyTorch 2.0+ and mocks optional dependencies to ensure a smooth run.*

## 📓 Alternative Inference (Kaggle)
We have provided our original production notebooks in the `notebooks/` directory.
- `notebooks/sglatrack-that-delivered.ipynb`: The notebook that generated our best public LB score.

## 🏋️ Training
To train the model from scratch (requires full tracking datasets like LaSOT, GOT-10k, etc.):
1. Configure your local paths in `src/lib/train/admin/local.py`.
2. Run the training script:
```bash
python src/tracking/train.py \
    --script sglatrack \
    --config deit_distilled \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 4 \
    --use_wandb 0
```

## 🧠 Methodology

Our solution is based on the original SGLATrack (Similarity-Guided Layer-Adaptive Transformer) framework for single-object visual tracking. The model uses transformer-based feature representations and adaptive similarity matching to maintain robust tracking performance across challenging UAV sequences.

### Base Tracker
We used the original SGLATrack implementation with the DeiT Distilled transformer backbone. The tracker was trained and evaluated using the official tracking pipeline while adapting the training setup for UAV-oriented tracking scenarios.

### Training Strategy
The model was trained on the UAV123 dataset to improve performance on aerial tracking sequences and motion patterns commonly found in drone footage. Training followed the original SGLATrack configuration and optimization pipeline with minimal architectural modifications.

### Experimental Modification
We experimented with integrating an object detection module into the tracking pipeline to recover the target when tracking confidence was lost or when the object temporarily disappeared from the scene. The goal was to improve long-term robustness and re-detection capability.

However, during validation and public leaderboard evaluation, this modification resulted in lower tracking performance compared to the baseline tracker. Due to the reduced public leaderboard score, the final submission uses the original SGLATrack inference pipeline without the additional detection module.

### Inference Pipeline
1. The tracker is initialized using the ground-truth bounding box from the first frame.
2. Frames are processed sequentially using the pretrained SGLATrack model.
3. The tracker predicts the target bounding box for every frame.
4. Predictions are exported in CSV format following the competition submission specification.

### Implementation Details
- Backbone: DeiT Distilled Transformer
- Tracker: Original SGLATrack
- Training Dataset: UAV123
- Framework: PyTorch
- Video Processing: OpenCV
- Output Format: Competition-compatible CSV submission file

Additional compatibility patches were implemented for PyTorch 2.0+ and optional dependency handling to ensure reproducible inference across different environments.

---

## 📁 Repository Structure
```
.
├── checkpoints/          # Model weights and download links
├── src/                  # Core library and tracking scripts
│   ├── lib/              # Model architecture, datasets, and utilities
│   ├── experiments/      # YAML configuration files
│   └── tracking/         # Training and testing entry points
├── notebooks/            # Original Kaggle notebooks and documentation
├── inference.py          # Standalone inference script for submission
├── requirements.txt      # Dependency list
├── .gitignore            # Git exclusion rules
└── README.md             # This file
```
