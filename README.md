## 🗺️ LLM Benchmark Pipeline Overview

### 1. Setup
- Creates **virtual environment** for isolated package management.
- Installs **required dependencies** automatically.

---

### 2. Model Loading
- Loads **each model** using the specified backend(s)  
  _Examples: `transformers`, `ollama`_

---

### 3. Benchmarking Loop
For **each** model, backend, and prompt length combination:
- Generates a **dynamic prompt** with the specified length.
- Runs **multiple inference passes** for consistency.
- Records **inference time** for each run.

---

### 4. Reporting
- Generates **detailed reports** in:
  - Text format
  - JSON format
  - Visualizations (charts/graphs for easy analysis)

---

### 5. Cleanup
- **Removes virtual environment** and any temporary files after benchmarking is complete.

---

## ⚙️ Requirements

- **Python 3.8+**
- **Sufficient disk space** (for downloaded models)
- **GPU** (optional, but recommended for faster inference)
- **Ollama installed and running** (if using the Ollama backend)

---

> _This pipeline is fully automated: all installations and environment management are handled by the script itself._


Basic Model Combinations
```
# Test all three default models with Transformers backend
python llm-benchmark.py --models phi3 mistral llama3 --backends transformers --word-counts 50 200 500 --runs 2

# Test all three default models with Ollama backend
python llm-benchmark.py --models phi3 mistral llama3 --backends ollama --word-counts 50 200 500 --runs 2

# Test all models with both backends
python llm-benchmark.py --models phi3 mistral llama3 --backends transformers ollama --word-counts 50 200 500 --runs 2
```

Additional Models
```
# Testing smaller models
python llm-benchmark.py --models phi3-mini tinyllama qwen2-1.5b --backends transformers --word-counts 50 200 500 --runs 2


All models

```
python llm-benchmark.py --models all --backends transformers --word-counts 50 100 --runs 1
```



# Testing larger models (requires more VRAM)
python llm-benchmark.py --models llama3-70b mistral-medium mixtral-8x7b --backends transformers --word-counts 100 500 --runs 1

# Mix of model sizes
python llm-benchmark.py --models phi3 mistral-7b gemma-7b falcon-7b --backends transformers --word-counts 100 500 --runs 2
```
