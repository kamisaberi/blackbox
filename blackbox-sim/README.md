Here is the standalone **README.
md** for the **`blackbox-sim`** module.

Place this file at **`blackbox/blackbox-sim/README.md`**.

***

```markdown
# 🧠 Blackbox Sim
### AI Research, Training, and Artifact Export Lab

[![Language](https://img.shields.io/badge/language-Python-blue)]()
[![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)]()
[![Inference](https://img.shields.io/badge/inference-ONNX%20%7C%20TensorRT-green)]()

**Blackbox Sim** is the offline R&D environment for the platform. It is responsible for training the Neural Networks that power the **Blackbox Core**. It takes raw text logs and converts them into the binary artifacts (`.plan`, `.onnx`, `vocab.txt`) required by the C++ engine.

This module is where all the AI/ML experimentation happens.

---

## ⚡ Key Architecture

The `Sim` is designed as a **Training-to-Inference Pipeline**.

1.  **Ingestion:** Reads raw data (CSV, TXT, PCAP) from `data/raw/`.
2.  **Preprocessing:** Tokenizes text, scales numerical features, and normalizes log templates using the **Drain** algorithm.
3.  **Training:** Trains models using **PyTorch** and a library of architectures (Autoencoders, Transformers, Mamba).
4.  **Export:** Converts the trained model to the **ONNX** (Open Neural Network Exchange) format, which acts as a bridge to high-performance inference engines like TensorRT.

---

## 📂 Project Structure

```text
blackbox-sim/
├── requirements.txt           # Python Dependencies
├── Dockerfile                 # CUDA-enabled environment
├── src/
│   ├── models/                # PyTorch model definitions (Autoencoder, Mamba)
│   ├── preprocessing/         # Tokenizer, Scaler, Log Parsers
│   ├── loss/                  # Custom loss functions (NT-Xent for SSL)
│   └── rl/                    # Reinforcement Learning environments
└── scripts/
    ├── train.py               # Main training script for log anomalies
    ├── train_network.py       # Training script for NetFlow data
    └── export_to_trt.sh       # Helper to convert ONNX -> TensorRT
```

---

## 🛠️ Setup & Usage

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU with CUDA 12.0+ (Highly Recommended)
*   Docker (for reproducible builds)

### 1. Docker-Based Training (Recommended)
This method ensures all CUDA dependencies are correctly configured.

```bash
# Build the training environment
docker build -t blackbox-sim .

# Run training, mounting your local data folders
docker run --rm -it --gpus all \
    -v $(pwd)/data:/app/data \
    blackbox-sim
```
*The entry point is `python scripts/train.py`, which will start training automatically.*

### 2. Local Training
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train.py
```

---

## 📦 Exported Artifacts

After a successful training run, the following files are generated in `data/artifacts/`. These are **critical** for `blackbox-core` to function.

| File | C++ Module | Purpose |
| :--- | :--- | :--- |
| `vocab.txt` | `Tokenizer` | Maps log words to integer IDs. |
| `scaler_params.txt` | `FeatureScaler`| Min/Max values for normalizing numerical inputs. |
| `autoencoder.onnx` | `InferenceEngine`| The neural network in a portable format. |
| `center_vec.txt` | `InferenceEngine`| (For SSL Models) The "normal" center point vector. |

### Converting ONNX to TensorRT

To get maximum speed in C++, convert the exported `.onnx` file to a `.plan` (TensorRT Engine). This is usually done on the target server.

```bash
# Run this inside the blackbox-core container
/usr/src/tensorrt/bin/trtexec \
  --onnx=/app/config/autoencoder.onnx \
  --saveEngine=/app/models/autoencoder.plan \
  --fp16 --int8
```

---

## 🧪 Included Models

The `Sim` includes pre-built architectures for:
*   **Autoencoder:** Simple, fast anomaly detection for numerical data.
*   **LogBERT/Transformer:** Semantic understanding of log text (via SSL).
*   **Mamba (SSM):** Long-context anomaly detection for time-series data.
*   **PPO (Reinforcement Learning):** For training the automated response agent.

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
```