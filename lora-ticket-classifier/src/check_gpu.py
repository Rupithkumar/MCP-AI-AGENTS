from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See the README, step 'Install PyTorch'.")
        return 1
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Training would run on the CPU and take hours.")
        print("      Install a CUDA build of PyTorch (pytorch.org selector) and update your NVIDIA driver.")
        return 1
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"gpu: {props.name}  memory: {props.total_memory / 2**30:.1f} GiB  compute capability: {major}.{minor}")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"bf16 supported: {bf16}")
    try:  
        dtype = torch.bfloat16 if bf16 else torch.float16
        a = torch.randn(512, 512, device="cuda", dtype=dtype)
        torch.cuda.synchronize()
        _ = (a @ a).float().sum().item()
    except Exception as exc:
        print(f"FAIL: the GPU could not run a test computation: {type(exc).__name__}: {str(exc)[:200]}")
        print("      Your PyTorch build probably does not support this GPU. Install a newer CUDA build.")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"free memory: {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")
    print("PASS: the GPU works. Suggested model: Qwen/Qwen2.5-0.5B-Instruct (about 1 GB in bf16).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
