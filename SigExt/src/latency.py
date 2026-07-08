import time
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer
from train_longformer_extractor_context import KeywordExtractorClf, MODELS 

def benchmark_model(model_key, checkpoint_path, device="cuda"):
    print(f"--- Benchmarking {model_key} ---")
    
    # 1. Load Model
    try:
        model_config = MODELS[model_key]
        model = KeywordExtractorClf.load_from_checkpoint(checkpoint_path, model_config=model_config)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return
    
    # 2. Prepare Dummy Input (Standardized to 4096 for fair comparison)
    max_len = 4096 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_len)).to(device)
    
    # BigBird Fix
    if model_config.get("needs_block_padding", False):
        block_size = model_config.get("block_size", 64)
        pad_len = block_size - (max_len % block_size)
        if pad_len != block_size:
            padding = torch.zeros((1, pad_len), dtype=torch.long).to(device)
            dummy_input = torch.cat([dummy_input, padding], dim=1)

    # 3. Measure Memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        _ = model.clf(dummy_input)
        
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
    print(f"Peak VRAM: {peak_memory:.2f} GB")

    # 4. Measure Latency (With Synchronization)
    latencies = []
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.clf(dummy_input)
            torch.cuda.synchronize() # Wait for GPU
            
    # Measurement
    for _ in range(50):
        torch.cuda.synchronize() # Sync before starting
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model.clf(dummy_input)
        
        torch.cuda.synchronize() # Sync after finishing (Crucial!)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # ms

    avg_latency = np.mean(latencies)
    throughput = 1000 / avg_latency # Docs per second
    
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Throughput:  {throughput:.2f} docs/sec")
    print("-----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--longformer_ckpt", type=str, required=True)
    parser.add_argument("--bigbird_ckpt", type=str, required=True)
    parser.add_argument("--modernbert_ckpt", type=str, required=True)
    args = parser.parse_args()

    benchmark_model("longformer_large", args.longformer_ckpt)
    benchmark_model("bigbird_large", args.bigbird_ckpt) 
    benchmark_model("modernbert_large", args.modernbert_ckpt)