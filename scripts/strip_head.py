import sys
import shutil
import os
from safetensors.torch import load_file, save_file

src_dir = sys.argv[1]
dst_dir = src_dir + "_stripped"

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

sf_path = os.path.join(dst_dir, "model.safetensors")
tensors = load_file(sf_path)

# Remove classifier head tensors to avoid llama.cpp name collisions
to_delete = [k for k in tensors.keys() if k.startswith("classifier.")]
for k in to_delete:
    print(f"Stripping {k} to prevent GGUF collision...")
    del tensors[k]

save_file(tensors, sf_path)
print(dst_dir)
