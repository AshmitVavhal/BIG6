import sys
import psutil
import torch

def check_environment():
    print("=" * 60)
    print("SATQUERY AI - HARDWARE & GPU DIAGNOSTIC SUITE")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        count = torch.cuda.device_count()
        print(f"GPU Count: {count}")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"  [GPU {i}] {name}")
            print(f"         Compute Capability: {props.major}.{props.minor}")
            print(f"         Total VRAM: {total_gb:.2f} GB")
            print(f"         Multiprocessors: {props.multi_processor_count}")
    else:
        print("  Running in CPU Fallback Mode. Model inference will execute on CPU.")

    vm = psutil.virtual_memory()
    print(f"System RAM: {vm.total / (1024**3):.1f} GB (Available: {vm.available / (1024**3):.1f} GB, {vm.percent}% used)")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print("=" * 60)

if __name__ == "__main__":
    check_environment()
