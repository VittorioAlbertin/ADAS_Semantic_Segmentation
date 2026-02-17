import torch
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model

def benchmark_model(model_name, device, input_size=(1, 3, 512, 1024), iterations=120, warmup=20, use_amp=False):
    precision_str = "AMP (FP16)" if use_amp else "FP32"
    print(f"Benchmarking {model_name} [{precision_str}]...")
    
    try:
        model = get_model(model_name, num_classes=19)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    model.to(device)
    model.eval()
    
    # Create dummy input
    input_tensor = torch.randn(*input_size).to(device)
    
    # Warmup
    print(f"  Warmup ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
                
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    # Benchmark
    print(f"  Running benchmark ({iterations} iters)...")
    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = time.time()
            else:
                start = time.time()
                
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                end = time.time()
            else:
                end = time.time()
            
            timings.append(end - start)
            
    mean_latency = np.mean(timings) * 1000 # ms
    std_latency = np.std(timings) * 1000 # ms
    fps = 1.0 / np.mean(timings)
    
    print(f"  Result: {mean_latency:.2f} ms ± {std_latency:.2f} ms | {fps:.2f} FPS")
    
    # Cleanup
    del model
    del input_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return {
        'model': model_name,
        'precision': precision_str,
        'latency_mean': mean_latency,
        'latency_std': std_latency,
        'fps': fps
    }

def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. AMP benchmark might not work as expected or will be slow on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        
    print(f"Using device: {device}")
    print(f"Input Resolution: 1024x2048")
    
    models_to_test = ['unet', 'deeplab', 'segformer']
    amp_modes = [False, True]
    results = []
    
    for model_name in models_to_test:
        for use_amp in amp_modes:
            # Skip AMP on CPU generally (though bfloat16 exists, let's stick to cuda amp for now)
            if device.type == 'cpu' and use_amp:
                continue
                
            res = benchmark_model(model_name, device, use_amp=use_amp)
            if res:
                results.append(res)
            
    print("\n" + "="*80)
    print("Final Inference Benchmark Results (Input: 1024x2048)")
    print("="*80)
    print(f"{'Model':<15} {'Precision':<12} {'Latency (ms)':<20} {'FPS':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<15} {res['precision']:<12} {res['latency_mean']:>7.2f} ± {res['latency_std']:<8.2f} {res['fps']:>8.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
