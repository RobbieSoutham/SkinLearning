import IPython
from gpustat import GPUStatCollection

"""
    Checks if code is being run from a notebook.
"""
def running_in_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:
            return False
    except:
        return False
    return True

def get_gpu_usage():
    gpu_stats = GPUStatCollection.new_query()
    print("\n")
    for gpu in gpu_stats:
        memory_used_gb = gpu.memory_used / 1024  # Convert from MB to GB
        memory_total_gb = gpu.memory_total / 1024  # Convert from MB to GB
        print(f"GPU {gpu.index}: {memory_used_gb:.2f} / {memory_total_gb:.2f} GB")
    print('\n')