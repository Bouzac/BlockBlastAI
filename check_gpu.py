import torch
import time

print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA dispo: {torch.cuda.is_available()}")
print(f"GPU:  {torch.cuda. get_device_name(0)}")
print("=" * 60)

size = 10000

# ===== TEST CPU =====
print("\n⏳ Test CPU en cours...")
a_cpu = torch. randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
for _ in range(5):
    c_cpu = a_cpu @ b_cpu
cpu_time = time. time() - start
print(f"✅ CPU: {cpu_time:.2f} sec")

# ===== TEST GPU =====
print("\n⏳ Test GPU en cours...  (regarde nvidia-smi! )")
a_gpu = torch.randn(size, size, device="cuda")
b_gpu = torch.randn(size, size, device="cuda")

torch.cuda.synchronize()
start = time.time()
for _ in range(5):
    c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"✅ GPU: {gpu_time:.2f} sec")

# ===== RÉSULTAT =====
print("\n" + "=" * 60)
if gpu_time < cpu_time:
    print(f"🚀 GPU est {cpu_time/gpu_time:.1f}x PLUS RAPIDE")
    print("✅ TON GPU FONCTIONNE CORRECTEMENT!")
else:
    print("❌ PROBLÈME:  GPU plus lent que CPU")
print(f"\n📊 VRAM utilisée:  {torch.cuda. memory_allocated()/1e9:.2f} GB")
print("=" * 60)