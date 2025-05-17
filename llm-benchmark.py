#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import venv
import random
import json
import shutil
import platform
import logging
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_benchmark.log")
    ]
)
logger = logging.getLogger('llm-benchmark')

class LLMBenchmark:
    def __init__(self, models=None, backends=None, venv_path="./llm_bench_venv", 
                 word_counts=None, num_runs=3, output_dir="./benchmark_results",
                 cuda_only=True, gpu_memory_limit=12):
        """Initialize the LLM Benchmarking tool."""
        self.models = models or ["phi3", "mistral", "llama3"]
        self.backends = backends or ["transformers", "ollama"]
        self.venv_path = Path(venv_path)
        self.word_counts = word_counts or [50, 100, 200, 500, 1000]
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.results = {}
        self.python_executable = None
        self.system_info = self._get_system_info()
        self.cuda_only = cuda_only
        self.gpu_memory_limit = gpu_memory_limit  # In GB
        self.log_file = self.output_dir / "detailed_benchmark.log"
        self.temp_dir = Path(tempfile.gettempdir()) / "llm_benchmark"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_system_info(self):
        """Collect system information for reporting."""
        gpu_info = self._get_gpu_info()
        
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "gpu": gpu_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_gpu_info(self):
        """Get GPU information using nvidia-smi."""
        try:
            # Run nvidia-smi to get GPU info
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 3:
                    return {
                        "name": gpu_info[0].strip(),
                        "memory_total": gpu_info[1].strip(),
                        "driver_version": gpu_info[2].strip()
                    }
            
            return {"error": "Could not get GPU information"}
        except:
            return {"error": "nvidia-smi not available"}
    
    def _check_cuda_availability(self):
        """Check if CUDA is available in the virtual environment."""
        if not self.cuda_only:
            return True
            
        check_script = """
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"CUDA is available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("CUDA is not available")
    exit(1)  # Exit with error code if CUDA is not available
"""
        
        temp_file = self.temp_dir / "check_cuda.py"
        with open(temp_file, "w") as f:
            f.write(check_script)
            
        cmd = [str(self.python_executable), str(temp_file)]
        
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            logger.error("CUDA is not available. Cannot continue with CUDA-only mode.")
            logger.error(f"Error: {process.stderr}")
            return False
            
        logger.info(process.stdout.strip())
        return True
        
    def create_venv(self):
        """Create a virtual environment for testing."""
        logger.info(f"Creating virtual environment at {self.venv_path}")
        
        if self.venv_path.exists():
            logger.warning(f"Virtual environment already exists at {self.venv_path}. Removing it.")
            shutil.rmtree(self.venv_path)
            
        venv.create(self.venv_path, with_pip=True)
        
        # Determine the Python executable path
        if os.name == 'nt':  # Windows
            self.python_executable = self.venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/MacOS
            self.python_executable = self.venv_path / "bin" / "python"
            
        logger.info(f"Using Python executable: {self.python_executable}")
        
        return self.python_executable
    
    def install_dependencies(self):
        """Install required packages in the virtual environment."""
        logger.info("Installing dependencies")
        
        base_packages = [
            "torch",
            "transformers", 
            "accelerate", 
            "bitsandbytes", 
            "scipy", 
            "numpy",
            "matplotlib", 
            "pandas", 
            "tabulate", 
            "tqdm",
            "sentencepiece",
            "protobuf",
            "einops",
            "psutil",
            "gputil",
            "py3nvml"
        ]
        
        backend_packages = {
            "transformers": [],
            "ollama": ["ollama"],
        }
        
        # Install PyTorch with CUDA
        if self.cuda_only:
            logger.info("Installing PyTorch with CUDA support")
            # Install PyTorch with CUDA (latest stable version)
            pytorch_cmd = [
                str(self.python_executable), 
                "-m", 
                "pip", 
                "install", 
                "torch", 
                "torchvision", 
                "torchaudio", 
                "--index-url", 
                "https://download.pytorch.org/whl/cu118"
            ]
            
            process = subprocess.run(
                pytorch_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Failed to install PyTorch with CUDA: {process.stderr}")
                raise RuntimeError(f"PyTorch installation failed: {process.stderr}")
        
        # Install other base packages
        self._run_pip_install([p for p in base_packages if p != "torch"])
        
        # Install backend-specific packages
        for backend in self.backends:
            if backend in backend_packages and backend_packages[backend]:
                self._run_pip_install(backend_packages[backend])
                
        # Check CUDA availability
        if self.cuda_only and not self._check_cuda_availability():
            raise RuntimeError("CUDA is required but not available. Aborting.")
            
        logger.info("All dependencies installed successfully")
    
    def _run_pip_install(self, packages):
        """Helper to run pip install commands."""
        if not packages:
            return
            
        cmd = [str(self.python_executable), "-m", "pip", "install"] + packages
        logger.info(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            logger.error(f"Failed to install packages: {process.stderr}")
            raise RuntimeError(f"Package installation failed: {process.stderr}")
            
        logger.debug(f"Pip install output: {process.stdout}")
    
    def generate_prompt(self, word_count):
        """Generate a prompt with the specified word count."""
        # Base text chunks of different lengths
        text_chunks = [
            "Explain the concept of machine learning and its applications in everyday life. ",
            "Describe how neural networks work and their relationship to biological neurons. ",
            "Discuss the ethical implications of artificial intelligence in modern society. ",
            "Compare and contrast supervised, unsupervised, and reinforcement learning approaches. ",
            "Analyze the potential impact of language models on the future of human-computer interaction. ",
            "Examine the role of data preprocessing in building effective machine learning models. ",
            "Elaborate on the challenges of bias in AI systems and potential solutions to mitigate them. ",
            "Describe the architecture of transformer models and why they're effective for NLP tasks. ",
            "Explain how generative models like GANs and VAEs create new content from training data. "
        ]
        
        prompt = ""
        current_word_count = 0
        
        # Keep adding chunks until we reach or exceed the desired word count
        while current_word_count < word_count:
            chunk = random.choice(text_chunks)
            prompt += chunk
            current_word_count = len(prompt.split())
        
        # Trim to exact word count if needed
        prompt_words = prompt.split()
        if len(prompt_words) > word_count:
            prompt = " ".join(prompt_words[:word_count])
            
        return prompt
    
    def _create_prompt_file(self, prompt):
        """Create a file containing the prompt with a short filename."""
        # Create a hash of the prompt to use as filename
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:10]
        prompt_file = self.temp_dir / f"prompt_{prompt_hash}.txt"
        
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
            
        return prompt_file
    
    def run_transformers_benchmark(self, model_name, prompt):
        """Run benchmark using HuggingFace Transformers."""
        # Create a file for the prompt instead of passing it directly
        prompt_file = self._create_prompt_file(prompt)
        
        # Create a temporary script to execute in the virtual environment
        temp_script = """
import time
import torch
import json
import logging
import psutil
import gc
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging for the subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('benchmark-subprocess')

def get_gpu_memory_usage():
    # Get current GPU memory usage
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
            "max_allocated": torch.cuda.max_memory_allocated(0) / (1024**3),
            "max_reserved": torch.cuda.max_memory_reserved(0) / (1024**3)
        }
        return gpu_memory
    return {"error": "CUDA not available"}

def get_model_size(model):
    # Get approximate size of model in GB.
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate approximate size (assuming float16 for most parameters)
    approx_size_gb = total_params * 2 / (1024**3)  # 2 bytes per parameter for float16
    return approx_size_gb

def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_model(model_name, prompt_file):
    # Read prompt from file to avoid command line length issues
    prompt = read_prompt_from_file(prompt_file)
    
    # Map generic model names to actual HF models
    model_mapping = {
        "deepseek-coder-1.3b-instruct": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-llm-7b-base": "deepseek-ai/deepseek-llm-7b-base",
        "facebook-opt-1.3b": "facebook/opt-1.3b",
        "codegemma-7b": "google/codegemma-7b",
        "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "microsoft-phi-2": "microsoft/phi-2",
        "microsoft-phi-3-mini-4k-instruct": "microsoft/phi-3-mini-4k-instruct",
        "microsoft-phi-3-mini-128k-instruct": "microsoft/phi-3-mini-128k-instruct",
        "mistral-7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
        "stability-stable-diffusion-3b": "stabilityai/stable-diffusion-3b",
        "thebloke-llama-2-7b-chat-gguf": "TheBloke/Llama-2-7B-Chat-GGUF",
        "tiiuae-falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
        "tinyllama-1.1b-chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft-phi-3": "microsoft/phi-3-mini-4k-instruct",  # alias for phi-3
        # Add any others as needed
    }

    
    hf_model_name = model_mapping.get(model_name.lower(), model_name)
    
    logger.info(f"Loading model: {hf_model_name}")
    
    if not torch.cuda.is_available():
        return {
            "status": "error",
            "error": "CUDA is not available"
        }
    
    device = "cuda"
    logger.info(f"Using device: {device}")
    
    # Collect system info before loading
    memory_before = psutil.virtual_memory()
    gpu_memory_before = get_gpu_memory_usage()
    logger.info(f"GPU memory before loading: {gpu_memory_before}")
    
    # Record start time for loading
    start_load = time.time()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Get model size
        model_size_gb = get_model_size(model)
        
        # Record load time
        load_time = time.time() - start_load
        
        # Collect system info after loading
        memory_after = psutil.virtual_memory()
        gpu_memory_after = get_gpu_memory_usage()
        
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        logger.info(f"Model size (approx): {model_size_gb:.2f} GB")
        logger.info(f"GPU memory after loading: {gpu_memory_after}")
        
        # Log memory increase
        gpu_memory_used = {k: gpu_memory_after[k] - gpu_memory_before[k] for k in gpu_memory_before if k in gpu_memory_after and "max" not in k}
        logger.info(f"GPU memory used by model: {gpu_memory_used}")
        
        # Warm-up run (important for more consistent benchmarking)
        logger.info("Performing warm-up run...")
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**input_ids, max_new_tokens=20)
        
        # Clearing cache after warm-up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Multiple timed runs for better statistics
        inference_times = []
        token_counts = []
        tokens_per_second = []
        
        num_runs = 3
        total_tokens = 100
        
        for i in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            
            # Record GPU memory before inference
            gpu_mem_before_inference = get_gpu_memory_usage()
            
            # Timed run
            logger.info(f"Starting inference run {i+1}/{num_runs}...")
            start_inference = time.time()
            
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **input_ids, 
                    max_new_tokens=total_tokens,
                    do_sample=False  # Deterministic for benchmarking
                )
                
            inference_time = time.time() - start_inference
            
            # Calculate tokens per second
            input_length = input_ids.input_ids.size(1)
            output_length = outputs.size(1)
            new_tokens = output_length - input_length
            tokens_per_sec = new_tokens / inference_time
            
            inference_times.append(inference_time)
            token_counts.append(new_tokens)
            tokens_per_second.append(tokens_per_sec)
            
            # GPU memory usage during inference
            gpu_mem_after_inference = get_gpu_memory_usage()
            gpu_mem_inference_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            
            logger.info(f"Run {i+1} completed in {inference_time:.4f}s | Tokens: {new_tokens} | Tokens/sec: {tokens_per_sec:.2f}")
            logger.info(f"Peak GPU memory during inference: {gpu_mem_inference_peak:.2f} GB")
        
        # Calculate average metrics
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_tokens_per_second = sum(tokens_per_second) / len(tokens_per_second)
        
        # Decode output for reference (last run)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Final memory cleanup
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "output": result,
            "load_time": load_time,
            "inference_time": avg_inference_time,
            "tokens_per_second": avg_tokens_per_second,
            "device": device,
            "model_size_gb": model_size_gb,
            "gpu_memory": {
                "before_load": gpu_memory_before,
                "after_load": gpu_memory_after,
                "model_usage": gpu_memory_used,
                "peak_inference": gpu_mem_inference_peak
            },
            "tokens_generated": sum(token_counts) / len(token_counts),
            "runs": num_runs
        }
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import sys
    
    model_name = sys.argv[1]
    prompt_file = sys.argv[2]
    
    try:
        result = run_model(model_name, prompt_file)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
"""
        
        # Use a short filename for the script
        script_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        temp_file = self.temp_dir / f"run_tf_{script_hash}.py"
        with open(temp_file, "w") as f:
            f.write(temp_script)
            
        # Execute the script in the virtual environment
        cmd = [
            str(self.python_executable),
            str(temp_file),
            model_name,
            str(prompt_file)
        ]
        
        try:
            logger.info(f"Running transformers benchmark for {model_name}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            # Log the stderr output regardless of success
            if process.stderr:
                logger.info(f"Stderr output for {model_name}:\n{process.stderr}")
            
            if process.returncode != 0:
                logger.error(f"Error running transformers benchmark: {process.stderr}")
                return {
                    "status": "error",
                    "error": process.stderr
                }
                
            # Save detailed output to log file
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / f"{model_name}_detailed_log.txt", "a") as f:
                f.write(f"\n\n--- Benchmark Run at {datetime.now().isoformat()} ---\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Prompt Length: {len(prompt.split())} words\n")
                f.write("STDOUT:\n")
                f.write(process.stdout)
                f.write("\nSTDERR:\n")
                f.write(process.stderr)
                
            # Extract the result JSON from the output
            output_lines = process.stdout.strip().split('\n')
            for line in reversed(output_lines):
                try:
                    result = json.loads(line)
                    if isinstance(result, dict) and "inference_time" in result:
                        # Add model name to result
                        result["model_name"] = model_name
                        result["timestamp"] = datetime.now().isoformat()
                        result["status"] = "success"
                        
                        # Save detailed metrics
                        with open(self.output_dir / f"{model_name}_metrics.json", "a") as f:
                            f.write(json.dumps(result, indent=2))
                            f.write("\n---\n")
                            
                        return result
                except json.JSONDecodeError:
                    continue
                    
            return {
                "status": "error",
                "error": "Could not parse benchmark results"
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out for {model_name}")
            return {
                "status": "error",
                "error": "Benchmark timed out"
            }
        finally:
            # Clean up temporary files
            try:
                os.remove(temp_file)
                os.remove(prompt_file)
            except:
                pass
            
    def run_ollama_benchmark(self, model_name, prompt):
        """Run benchmark using Ollama."""
        # Create a file for the prompt instead of passing it directly
        prompt_file = self._create_prompt_file(prompt)
        
        # Create a temporary script to execute in the virtual environment
        temp_script = """
import time
import json
import requests
import logging
import torch
import os

# Configure logging for the subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('benchmark-subprocess')

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
            "max_allocated": torch.cuda.max_memory_allocated(0) / (1024**3)
        }
        return gpu_memory
    return {"error": "CUDA not available"}

def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_model(model_name, prompt_file):
    # Read prompt from file to avoid command line length issues
    prompt = read_prompt_from_file(prompt_file)
    
    model_mapping = {
        "phi3": "phi3",
        "mistral": "mistral",
        "llama3": "llama3",
        "llama3-8b": "llama3:8b",
        "mistral-7b": "mistral:7b-instruct-v0.2",
        # Add more mappings as needed
    }
    
    if not torch.cuda.is_available():
        return {
            "status": "error",
            "error": "CUDA is not available"
        }
    
    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()
    gpu_memory_before = get_gpu_memory_usage()
    logger.info(f"GPU memory before Ollama run: {gpu_memory_before}")
    
    ollama_model = model_mapping.get(model_name.lower(), model_name)
    
    logger.info(f"Using Ollama model: {ollama_model}")
    
    # First, make sure model is pulled/downloaded (this is the "load" time)
    start_load = time.time()
    
    try:
        # Send request with CUDA acceleration flag
        load_response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": ollama_model, "stream": False}
        )
        
        if load_response.status_code != 200:
            return {
                "status": "error",
                "error": f"Failed to pull model: {load_response.text}"
            }
            
        load_time = time.time() - start_load
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Get the model info to determine size
        model_info_response = requests.get(f"http://localhost:11434/api/show?name={ollama_model}")
        model_info = model_info_response.json() if model_info_response.status_code == 200 else {}
        model_size_mb = model_info.get("size", 0) / (1024 * 1024)  # Convert bytes to MB
        model_size_gb = model_size_mb / 1024  # Convert MB to GB
        
        logger.info(f"Model size: {model_size_gb:.2f} GB")
        
        # Warm-up run
        logger.info("Performing warm-up run...")
        warm_up_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ollama_model, 
                "prompt": "Hello, world!", 
                "stream": False,
                "options": {
                    "num_gpu": 99  # Use all available GPUs
                }
            }
        )
        
        # Reset GPU memory stats after warm-up
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_after_warmup = get_gpu_memory_usage()
        logger.info(f"GPU memory after warm-up: {gpu_memory_after_warmup}")
        
        # Multiple timed runs for better statistics
        inference_times = []
        token_counts = []
        tokens_per_second = []
        
        num_runs = 3
        
        for i in range(num_runs):
            logger.info(f"Starting inference run {i+1}/{num_runs}...")
            start_inference = time.time()
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": ollama_model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "num_gpu": 99,  # Use all available GPUs
                        "num_thread": 0  # Auto-detect number of threads
                    }
                }
            )
            
            inference_time = time.time() - start_inference
            
            if response.status_code != 200:
                logger.error(f"Inference failed: {response.text}")
                continue
                
            result = response.json()
            
            # Get token count and calculate tokens per second
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0) / 1000000000  # Convert nanoseconds to seconds
            tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
            
            inference_times.append(inference_time)
            token_counts.append(eval_count)
            tokens_per_second.append(tokens_per_sec)
            
            # GPU memory usage during inference
            gpu_memory_after = get_gpu_memory_usage()
            gpu_mem_inference_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            
            logger.info(f"Run {i+1} completed in {inference_time:.4f}s | Tokens: {eval_count} | Tokens/sec: {tokens_per_sec:.2f}")
            logger.info(f"Peak GPU memory during inference: {gpu_mem_inference_peak:.2f} GB")
        
        # Calculate average metrics
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_tokens_per_second = sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        return {
            "status": "success",
            "output": result.get("response", ""),
            "load_time": load_time,
            "inference_time": avg_inference_time,
            "tokens_per_second": avg_tokens_per_second,
            "tokens_generated": avg_tokens,
            "device": "ollama-gpu",
            "model_size_gb": model_size_gb,
            "gpu_memory": {
                "before_run": gpu_memory_before,
                "after_warmup": gpu_memory_after_warmup,
                "peak_inference": gpu_mem_inference_peak
            },
            "runs": num_runs
        }
    except Exception as e:
        logger.error(f"Error in Ollama benchmark: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import sys
    
    model_name = sys.argv[1]
    prompt_file = sys.argv[2]
    
    try:
        result = run_model(model_name, prompt_file)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
"""
        
        # Use a short filename for the script
        script_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        temp_file = self.temp_dir / f"run_ol_{script_hash}.py"
        with open(temp_file, "w") as f:
            f.write(temp_script)
            
        # Execute the script in the virtual environment
        cmd = [
            str(self.python_executable),
            str(temp_file),
            model_name,
            str(prompt_file)
        ]
        
        try:
            logger.info(f"Running Ollama benchmark for {model_name}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            # Log the stderr output regardless of success
            if process.stderr:
                logger.info(f"Stderr output for {model_name} (Ollama):\n{process.stderr}")
            
            if process.returncode != 0:
                logger.error(f"Error running Ollama benchmark: {process.stderr}")
                return {
                    "status": "error",
                    "error": process.stderr
                }
                
            # Save detailed output to log file
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / f"{model_name}_ollama_detailed_log.txt", "a") as f:
                f.write(f"\n\n--- Benchmark Run at {datetime.now().isoformat()} ---\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Prompt Length: {len(prompt.split())} words\n")
                f.write("STDOUT:\n")
                f.write(process.stdout)
                f.write("\nSTDERR:\n")
                f.write(process.stderr)
                
            # Extract the result JSON from the output
            output_lines = process.stdout.strip().split('\n')
            for line in reversed(output_lines):
                try:
                    result = json.loads(line)
                    if isinstance(result, dict) and "inference_time" in result:
                        # Add model name to result
                        result["model_name"] = model_name
                        result["backend"] = "ollama"
                        result["timestamp"] = datetime.now().isoformat()
                        result["status"] = "success"
                        
                        # Save detailed metrics
                        with open(self.output_dir / f"{model_name}_ollama_metrics.json", "a") as f:
                            f.write(json.dumps(result, indent=2))
                            f.write("\n---\n")
                            
                        return result
                except json.JSONDecodeError:
                    continue
                    
            return {
                "status": "error",
                "error": "Could not parse benchmark results"
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out for {model_name}")
            return {
                "status": "error",
                "error": "Benchmark timed out"
            }
        finally:
            # Clean up temporary files
            try:
                os.remove(temp_file)
                os.remove(prompt_file)
            except:
                pass
    
    def run_benchmarks(self):
        """Run benchmarks for all models, backends, and word counts."""
        logger.info("Starting benchmark runs")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for model in self.models:
            self.results[model] = {}
            
            for backend in self.backends:
                self.results[model][backend] = {}
                
                for word_count in self.word_counts:
                    self.results[model][backend][word_count] = []
                    
                    for run in range(self.num_runs):
                        logger.info(f"Run {run+1}/{self.num_runs} - Model: {model}, Backend: {backend}, Words: {word_count}")
                        
                        # Generate prompt
                        prompt = self.generate_prompt(word_count)
                        
                        # Run the appropriate benchmark
                        if backend == "transformers":
                            result = self.run_transformers_benchmark(model, prompt)
                        elif backend == "ollama":
                            result = self.run_ollama_benchmark(model, prompt)
                        else:
                            logger.error(f"Unknown backend: {backend}")
                            result = {"status": "error", "error": f"Unknown backend: {backend}"}
                            
                        # Record results
                        if result["status"] == "success":
                            self.results[model][backend][word_count].append({
                                "run": run + 1,
                                "inference_time": result["inference_time"],
                                "load_time": result.get("load_time", None),
                                "device": result.get("device", "unknown"),
                                "status": "success"
                            })
                        else:
                            self.results[model][backend][word_count].append({
                                "run": run + 1,
                                "error": result.get("error", "Unknown error"),
                                "status": "error"
                            })
                            
                        # Output run results immediately
                        self.output_run_results(model, backend, word_count, run + 1, result)
        
        # Generate final report
        self.generate_report()
        
        return self.results
    
    def output_run_results(self, model, backend, word_count, run, result):
        """Output results of a single benchmark run."""
        if result["status"] == "success":
            logger.info(f"Run {run} completed - Inference Time: {result['inference_time']:.4f}s")
        else:
            logger.error(f"Run {run} failed - Error: {result.get('error', 'Unknown error')}")
    
    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        logger.info("Generating benchmark report")
        
        # Create table of average inference times
        headers = ["Model", "Backend"] + [f"{wc} words" for wc in self.word_counts] + ["Tokens/sec", "GPU Mem (GB)"]
        table_data = []
        
        for model in self.models:
            for backend in self.backends:
                row = [model, backend]
                
                gpu_memory_usage = []  # Track memory usage across word counts
                tokens_per_second = []  # Track tokens/sec across word counts
                
                for word_count in self.word_counts:
                    # Get successful runs
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        # Calculate average inference time
                        avg_time = sum(run["inference_time"] for run in successful_runs) / len(successful_runs)
                        row.append(f"{avg_time:.4f}s")
                        
                        # Collect GPU memory and tokens/sec stats for averages
                        for run in successful_runs:
                            if "gpu_memory" in run and "peak_inference" in run["gpu_memory"]:
                                gpu_memory_usage.append(run["gpu_memory"]["peak_inference"])
                            if "tokens_per_second" in run:
                                tokens_per_second.append(run["tokens_per_second"])
                    else:
                        row.append("N/A")
                
                # Add tokens per second and GPU memory columns
                avg_tokens_per_sec = sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0
                avg_gpu_memory = sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0
                
                row.append(f"{avg_tokens_per_sec:.2f}")
                row.append(f"{avg_gpu_memory:.2f}")
                
                table_data.append(row)
                
        # Generate table
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        
        # Create GPU memory usage table
        gpu_headers = ["Model", "Backend", "Model Size (GB)", "Load Memory (GB)", "Inference Peak (GB)"]
        gpu_table_data = []
        
        for model in self.models:
            for backend in self.backends:
                # Find a successful run to get GPU stats
                model_size = "N/A"
                load_memory = "N/A"
                inference_peak = "N/A"
                
                for word_count in self.word_counts:
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs and "model_size_gb" in successful_runs[0]:
                        model_size = f"{successful_runs[0]['model_size_gb']:.2f}"
                        
                        if "gpu_memory" in successful_runs[0]:
                            gpu_memory = successful_runs[0]["gpu_memory"]
                            
                            if backend == "transformers" and "after_load" in gpu_memory and "allocated" in gpu_memory["after_load"]:
                                load_memory = f"{gpu_memory['after_load']['allocated']:.2f}"
                            elif backend == "ollama" and "after_warmup" in gpu_memory and "allocated" in gpu_memory["after_warmup"]:
                                load_memory = f"{gpu_memory['after_warmup']['allocated']:.2f}"
                                
                            if "peak_inference" in gpu_memory:
                                inference_peak = f"{gpu_memory['peak_inference']:.2f}"
                        
                        break
                
                gpu_table_data.append([model, backend, model_size, load_memory, inference_peak])
        
        # Generate GPU table
        gpu_table = tabulate(gpu_table_data, headers=gpu_headers, tablefmt="grid")
        
        # Create summary info
        summary = {
            "system_info": self.system_info,
            "best_performers": self._get_best_performers(),
            "model_gpu_metrics": {model: self._get_model_gpu_metrics(model) for model in self.models},
            "raw_results": self.results
        }
        
        # Write report files
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Write text report
        text_report = f"""
# LLM Benchmark Report (CUDA-only)
Generated on: {datetime.now().isoformat()}

## System Information
- OS: {self.system_info['os']} {self.system_info['os_version']}
- Python: {self.system_info['python_version']}
- CPU: {self.system_info['cpu']}
- GPU: {self.system_info['gpu'].get('name', 'Unknown')} ({self.system_info['gpu'].get('memory_total', 'Unknown')})
- Driver: {self.system_info['gpu'].get('driver_version', 'Unknown')}

## Benchmark Results (Average Inference Time)
{table}

## GPU Memory Usage
{gpu_table}

## Best Performers
"""
        
        best_performers = self._get_best_performers()
        for word_count, best in best_performers.items():
            text_report += f"- {word_count} words: {best['model']} ({best['backend']}) - {best['time']:.4f}s | {best['tokens_per_second']:.2f} tokens/sec\n"
            
        # Add best overall model
        best_overall = self._get_best_overall_model()
        if best_overall:
            text_report += f"\n## Overall Best Model\n{best_overall['model']} ({best_overall['backend']}) - Avg {best_overall['avg_time']:.4f}s | {best_overall['avg_tokens_per_second']:.2f} tokens/sec\n"
        
        # Add recommendations
        text_report += "\n## Recommendations\n"
        text_report += self._generate_recommendations()
            
        with open(self.output_dir / f"report_{report_time}.txt", "w") as f:
            f.write(text_report)
            
        # Write JSON report with full details
        with open(self.output_dir / f"report_{report_time}.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        # Generate plots
        self._generate_plots(report_time)
        
        logger.info(f"Report generated: {self.output_dir}/report_{report_time}.txt")
        
        return text_report
    
    def _get_best_performers(self):
        """Get the best performing model/backend for each word count."""
        best_performers = {}
        
        for word_count in self.word_counts:
            best_time = float('inf')
            best_model = None
            best_backend = None
            best_tokens_per_second = 0
            
            for model in self.models:
                for backend in self.backends:
                    # Get successful runs
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        avg_time = sum(run["inference_time"] for run in successful_runs) / len(successful_runs)
                        
                        # Calculate tokens per second if available
                        tokens_per_second = 0
                        tokens_per_second_values = [run.get("tokens_per_second", 0) for run in successful_runs]
                        if tokens_per_second_values:
                            tokens_per_second = sum(tokens_per_second_values) / len(tokens_per_second_values)
                        
                        if avg_time < best_time:
                            best_time = avg_time
                            best_model = model
                            best_backend = backend
                            best_tokens_per_second = tokens_per_second
                            
            if best_model:
                best_performers[word_count] = {
                    "model": best_model,
                    "backend": best_backend,
                    "time": best_time,
                    "tokens_per_second": best_tokens_per_second
                }
                
        return best_performers
    
    def _get_best_overall_model(self):
        """Calculate the best overall model across all word counts."""
        model_scores = {}
        
        for model in self.models:
            for backend in self.backends:
                key = f"{model}_{backend}"
                model_scores[key] = {
                    "model": model,
                    "backend": backend,
                    "times": [],
                    "tokens_per_second": []
                }
                
                for word_count in self.word_counts:
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        avg_time = sum(run["inference_time"] for run in successful_runs) / len(successful_runs)
                        model_scores[key]["times"].append(avg_time)
                        
                        # Calculate tokens per second if available
                        tokens_per_second_values = [run.get("tokens_per_second", 0) for run in successful_runs]
                        if tokens_per_second_values:
                            avg_tokens_per_second = sum(tokens_per_second_values) / len(tokens_per_second_values)
                            model_scores[key]["tokens_per_second"].append(avg_tokens_per_second)
        
        # Find the model with the best average time across all word counts
        best_overall = None
        best_avg_time = float('inf')
        
        for key, score in model_scores.items():
            if score["times"]:
                avg_time = sum(score["times"]) / len(score["times"])
                avg_tokens_per_second = sum(score["tokens_per_second"]) / len(score["tokens_per_second"]) if score["tokens_per_second"] else 0
                
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_overall = {
                        "model": score["model"],
                        "backend": score["backend"],
                        "avg_time": avg_time,
                        "avg_tokens_per_second": avg_tokens_per_second
                    }
        
        return best_overall
    
    def _get_model_gpu_metrics(self, model_name):
        """Get GPU metrics for a specific model."""
        metrics = {}
        
        for backend in self.backends:
            metrics[backend] = {}
            
            # Find a successful run with GPU metrics
            for word_count in self.word_counts:
                successful_runs = [
                    run for run in self.results[model_name][backend][word_count] 
                    if run["status"] == "success" and "gpu_memory" in run
                ]
                
                if successful_runs:
                    metrics[backend] = {
                        "model_size_gb": successful_runs[0].get("model_size_gb", "N/A"),
                        "gpu_memory": successful_runs[0].get("gpu_memory", {}),
                        "tokens_per_second": successful_runs[0].get("tokens_per_second", 0)
                    }
                    break
        
        return metrics
    
    def _generate_recommendations(self):
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Find the best overall model
        best_overall = self._get_best_overall_model()
        if best_overall:
            recommendations.append(f"• Best overall model for your 12GB GPU: {best_overall['model']} with {best_overall['backend']} backend")
        
        # Find the most memory-efficient model
        memory_efficient_model = None
        lowest_memory_usage = float('inf')
        
        for model in self.models:
            for backend in self.backends:
                for word_count in self.word_counts:
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success" and "gpu_memory" in run and "peak_inference" in run["gpu_memory"]
                    ]
                    
                    if successful_runs:
                        memory_usage = successful_runs[0]["gpu_memory"]["peak_inference"]
                        if memory_usage < lowest_memory_usage:
                            lowest_memory_usage = memory_usage
                            memory_efficient_model = {
                                "model": model,
                                "backend": backend,
                                "memory_usage": memory_usage
                            }
        
        if memory_efficient_model:
            recommendations.append(f"• Most memory-efficient model: {memory_efficient_model['model']} with {memory_efficient_model['backend']} backend ({memory_efficient_model['memory_usage']:.2f} GB)")
        
        # Find the fastest model for short prompts
        short_prompt_models = self._get_best_performers().get(min(self.word_counts))
        if short_prompt_models:
            recommendations.append(f"• Best for short prompts: {short_prompt_models['model']} with {short_prompt_models['backend']} backend")
        
        # Find the fastest model for long prompts
        long_prompt_models = self._get_best_performers().get(max(self.word_counts))
        if long_prompt_models:
            recommendations.append(f"• Best for long prompts: {long_prompt_models['model']} with {long_prompt_models['backend']} backend")
        
        # Backend recommendations
        transformers_vs_ollama = {}
        for model in self.models:
            transformers_times = []
            ollama_times = []
            
            for word_count in self.word_counts:
                if "transformers" in self.backends:
                    successful_runs = [
                        run["inference_time"] for run in self.results[model]["transformers"][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        transformers_times.append(sum(successful_runs) / len(successful_runs))
                
                if "ollama" in self.backends:
                    successful_runs = [
                        run["inference_time"] for run in self.results[model]["ollama"][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        ollama_times.append(sum(successful_runs) / len(successful_runs))
            
            if transformers_times and ollama_times:
                avg_transformers = sum(transformers_times) / len(transformers_times)
                avg_ollama = sum(ollama_times) / len(ollama_times)
                
                transformers_vs_ollama[model] = {
                    "transformers": avg_transformers,
                    "ollama": avg_ollama,
                    "diff_percent": ((avg_ollama - avg_transformers) / avg_transformers) * 100
                }
        
        backend_insights = []
        for model, comparison in transformers_vs_ollama.items():
            if comparison["diff_percent"] > 10:
                backend_insights.append(f"• {model}: Transformers is {abs(comparison['diff_percent']):.1f}% faster than Ollama")
            elif comparison["diff_percent"] < -10:
                backend_insights.append(f"• {model}: Ollama is {abs(comparison['diff_percent']):.1f}% faster than Transformers")
        
        if backend_insights:
            recommendations.append("\nBackend performance comparisons:")
            recommendations.extend(backend_insights)
        
        return "\n".join(recommendations)
    
    def _generate_plots(self, report_time):
        """Generate plots visualizing the benchmark results."""
        # Plot 1: Inference time by word count for each model/backend
        plt.figure(figsize=(12, 8))
        
        for model in self.models:
            for backend in self.backends:
                x = []
                y = []
                
                for word_count in self.word_counts:
                    # Get successful runs
                    successful_runs = [
                        run["inference_time"] 
                        for run in self.results[model][backend][word_count] 
                        if run["status"] == "success"
                    ]
                    
                    if successful_runs:
                        x.append(word_count)
                        y.append(sum(successful_runs) / len(successful_runs))
                
                if x and y:
                    plt.plot(x, y, marker='o', label=f"{model} ({backend})")
        
        plt.xlabel('Prompt Length (words)')
        plt.ylabel('Average Inference Time (seconds)')
        plt.title('LLM Inference Time by Prompt Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / f"inference_time_plot_{report_time}.png")
        
        # Plot 2: Bar chart comparison for specific word count
        medium_word_count = self.word_counts[len(self.word_counts) // 2]
        
        labels = []
        times = []
        
        for model in self.models:
            for backend in self.backends:
                # Get successful runs
                successful_runs = [
                    run["inference_time"] 
                    for run in self.results[model][backend][medium_word_count] 
                    if run["status"] == "success"
                ]
                
                if successful_runs:
                    labels.append(f"{model}\n({backend})")
                    times.append(sum(successful_runs) / len(successful_runs))
        
        if labels and times:
            plt.figure(figsize=(12, 6))
            plt.bar(labels, times)
            plt.xlabel('Model (Backend)')
            plt.ylabel('Average Inference Time (seconds)')
            plt.title(f'LLM Inference Time Comparison ({medium_word_count} words)')
            plt.grid(True, axis='y')
            plt.savefig(self.output_dir / f"model_comparison_plot_{report_time}.png")
            
        # Plot 3: GPU Memory Usage
        labels = []
        memory_usage = []
        
        for model in self.models:
            for backend in self.backends:
                # Find a successful run with GPU metrics
                for word_count in self.word_counts:
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success" and "gpu_memory" in run and "peak_inference" in run["gpu_memory"]
                    ]
                    
                    if successful_runs:
                        labels.append(f"{model}\n({backend})")
                        memory_usage.append(successful_runs[0]["gpu_memory"]["peak_inference"])
                        break
        
        if labels and memory_usage:
            plt.figure(figsize=(12, 6))
            plt.bar(labels, memory_usage)
            plt.xlabel('Model (Backend)')
            plt.ylabel('GPU Memory Usage (GB)')
            plt.title('Peak GPU Memory Usage During Inference')
            plt.grid(True, axis='y')
            plt.savefig(self.output_dir / f"gpu_memory_plot_{report_time}.png")
            
        # Plot 4: Tokens per second comparison
        labels = []
        tokens_per_second = []
        
        for model in self.models:
            for backend in self.backends:
                # Find a successful run with tokens per second metrics
                for word_count in self.word_counts:
                    successful_runs = [
                        run for run in self.results[model][backend][word_count] 
                        if run["status"] == "success" and "tokens_per_second" in run
                    ]
                    
                    if successful_runs:
                        labels.append(f"{model}\n({backend})")
                        tokens_per_second.append(successful_runs[0]["tokens_per_second"])
                        break
        
        if labels and tokens_per_second:
            plt.figure(figsize=(12, 6))
            plt.bar(labels, tokens_per_second)
            plt.xlabel('Model (Backend)')
            plt.ylabel('Tokens per Second')
            plt.title('LLM Generation Speed (Tokens/sec)')
            plt.grid(True, axis='y')
            plt.savefig(self.output_dir / f"tokens_per_second_plot_{report_time}.png")
    
    def cleanup(self):
        """Clean up the virtual environment."""
        logger.info("Cleaning up")
        
        if self.venv_path.exists():
            logger.info(f"Removing virtual environment at {self.venv_path}")
            shutil.rmtree(self.venv_path)
        
        # Clean up temp directory 
        if self.temp_dir.exists():
            logger.info(f"Removing temporary files at {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            
        logger.info("Cleanup completed")
    
    def run(self):
        """Run the full benchmark process."""
        try:
            logger.info("Starting LLM benchmark")
            
            # Enable Windows long path support if on Windows
            if platform.system() == "Windows":
                logger.info("Enabling Windows long path support for this process")
                # Not directly possible from Python, but we'll handle it by using shorter paths
            
            self.create_venv()
            self.install_dependencies()
            self.run_benchmarks()
            
            logger.info("Benchmark completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
            return False
            
        finally:
            self.cleanup()

def main():
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(description="LLM Benchmarking Tool")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Models to benchmark (default: phi3, mistral, llama3). Use 'all' to benchmark all available models."
    )
    
    parser.add_argument(
        "--backends", 
        nargs="+", 
        help="Backends to use (default: transformers, ollama)"
    )
    
    parser.add_argument(
        "--word-counts", 
        type=int, 
        nargs="+", 
        help="Word counts for prompts (default: 50, 100, 200, 500, 1000)"
    )
    
    parser.add_argument(
        "--runs", 
        type=int, 
        default=3, 
        help="Number of runs per configuration (default: 3)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="./benchmark_results", 
        help="Directory for output files (default: ./benchmark_results)"
    )
    
    parser.add_argument(
        "--venv-path", 
        default="./llm_bench_venv", 
        help="Path for virtual environment (default: ./llm_bench_venv)"
    )
    
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Only run on CUDA (GPU), fail if CUDA is not available"
    )
    
    parser.add_argument(
        "--gpu-memory-limit",
        type=int,
        default=12,
        help="Maximum GPU memory in GB (default: 12)"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' for models
    if args.models and 'all' in args.models:
        # List of all transformers models
        transformers_models = [
             "deepseek-coder-1.3b-instruct", 
            # "deepseek-coder-6.7b-instruct",
            # "deepseek-llm-7b-base", 
             "facebook-opt-1.3b", 
            # "codegemma-7b",
            "llama-2-7b-chat-hf", 
            # "microsoft-phi-2",
            # "microsoft-phi-3-mini-4k-instruct", 
            "microsoft-phi-3-mini-128k-instruct",
            "mistral-7b-instruct-v0.1", 
            # "sentence-transformers-all-minilm-l6-v2",
            # "sentence-transformers-mpnet-base-v2", 
            # "stability-stable-diffusion-3b", 
            # "ts-small",
            # "thebloke-llama-2-7b-chat-gguf", 
            # "tiiuae-falcon-7b-instruct", 
            # "tinyllama-1.1b-chat-v1.0",
            # "microsoft-phi-3"
        ]
        
        # List of all ollama models
        ollama_models = [
            "phi3", 
            "mistral", 
            "llama3", 
            "llama3-8b", 
            "mistral-7b"
        ]
        
        # Combine models based on selected backends
        all_models = []
        if not args.backends or "transformers" in args.backends:
            all_models.extend(transformers_models)
        if not args.backends or "ollama" in args.backends:
            all_models.extend(ollama_models)
            
        args.models = all_models
        logger.info(f"Using 'all' option. Benchmarking {len(all_models)} models.")
        logger.info(f"Models to benchmark: {', '.join(all_models)}")
    
    benchmark = LLMBenchmark(
        models=args.models,
        backends=args.backends,
        word_counts=args.word_counts,
        num_runs=args.runs,
        output_dir=args.output_dir,
        venv_path=args.venv_path,
        cuda_only=args.cuda_only or True,  # Force CUDA-only mode
        gpu_memory_limit=args.gpu_memory_limit
    )
    
    success = benchmark.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())