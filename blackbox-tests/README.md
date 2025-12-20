Here is the standalone **README.md** for the **`blackbox-tests`** module.

Place this file at **`blackbox/blackbox-tests/README.md`**.

***

```markdown
# 🧪 Blackbox Tests
### Unified Quality Assurance & Integration Suite

[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Go%20%7C%20Python-purple)]()
[![Framework](https://img.shields.io/badge/framework-GoogleTest%20%7C%20PyTest-green)]()
[![Coverage](https://img.shields.io/badge/coverage-Unit%20%26%20Integration-blue)]()

**Blackbox Tests** is the centralized testing hub for the entire Blackbox ecosystem. Instead of burying tests inside individual module directories, this standalone module aggregates verification logic for the C++ Core, Go API, and Python AI labs.

It features a **Standalone CMake Build System** that compiles C++ unit tests independently from the main application binary, ensuring isolation and faster compile-test cycles.

---

## ⚡ Test Scope

| Layer | Framework | What is tested? |
| :--- | :--- | :--- |
| **Core (C++)** | GoogleTest (GTest) | Memory safety (RingBuffer), Parsing logic, Token Bucket algorithms, SQL generation. |
| **Tower (Go)** | Go Test (`testing`) | API endpoint responses, Config loading, JSON marshalling. |
| **Sim (Python)** | PyTest | Neural Network tensor shapes, Normalization math, Loss functions. |

---

## 🛠️ Prerequisites

To run the full suite, you need the toolchains for all three languages installed:

*   **C++:** CMake 3.20+, GCC/Clang
*   **Go:** Go 1.21+
*   **Python:** Python 3.10+ (with `torch` installed)

---

## 🚀 Running Tests

### 1. The Master Script (Recommended)
We provide a unified script that detects which tools are installed and runs all available tests in sequence.

```bash
./run_all.sh
```

### 2. C++ Unit Tests (Manual)
The C++ tests compile against the source files in `../blackbox-core/src`.

```bash
cd core
mkdir build && cd build
cmake ..
make -j$(nproc)
./run_core_tests
```

### 3. Go API Tests (Manual)
Verifies the Tower API logic.

```bash
cd tower
go test -v ./...
```

### 4. Python Logic Tests (Manual)
Verifies the AI math.

```bash
# Ensure PYTHONPATH includes the sim directory
export PYTHONPATH=$PYTHONPATH:../../blackbox-sim
pytest sim/
```

---

## 📂 Project Structure

```text
blackbox-tests/
├── run_all.sh                 # Master execution script
├── core/                      # C++ Tests
│   ├── CMakeLists.txt         # Standalone build config
│   ├── main.cpp               # GTest Entry Point
│   ├── ingest/
│   │   └── test_ring_buffer.cpp
│   ├── parser/
│   │   └── test_parser_engine.cpp
│   └── storage/
│       └── test_clickhouse_client.cpp
├── tower/                     # Go Tests
│   ├── main_test.go
│   └── api/
│       └── health_test.go
└── sim/                       # Python Tests
    └── models/
        └── test_autoencoder.py
```

---

## 🧪 Key Test Scenarios

### Ring Buffer Safety (C++)
*   **Overflow:** Verifies that pushing to a full buffer returns false without corrupting memory.
*   **FIFO Order:** Ensures logs come out in the exact order they went in.

### Tokenizer Consistency (C++ vs Python)
*   Verifies that the C++ `Tokenizer` implementation produces the exact same integer vectors as the Python `LogTokenizer` used during training. This is critical for AI accuracy.

### API Contract (Go)
*   Ensures that `/api/v1/health` returns `200 OK`.
*   Verifies that environment variables correctly override default configurations.

---

## 📄 License

**Proprietary & Confidential.**
Copyright © 2025 Ignition AI. All Rights Reserved.
```