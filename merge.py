import os
from safetensors import safe_open
from safetensors.torch import save_file

def merge_safetensor_files(sftsr_files, output_file="model.safetensors"):
    tensors = {}
    for file in sftsr_files:
        with safe_open(file, framework="pt") as sf_tsr:
            metadata = sf_tsr.metadata()
            for layer in sf_tsr.keys():
                blk_tensor = sf_tsr.get_tensor(str(layer))
                tensors[str(layer)] = blk_tensor
    
    save_file(tensors, output_file, metadata)

def get_safetensor_files(directory):
    """
    Retrieve all `.safetensors` files within a directory.
    Returns:
        list: A list of paths to the found `.safetensors` files.
    """
    safetensors_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_files.append(os.path.join(root, file))
    return safetensors_files

if __name__ == "__main__":
    safetensor_files = get_safetensor_files("./shards")
    print(f"The following shards/chunks will be merged : {safetensor_files}")
    
    merge_safetensor_files(safetensor_files, output_file="./output/merged_model.safetensors")
