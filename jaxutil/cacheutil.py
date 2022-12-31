
import gc
import psutil
import sys

def clear_caches():
    # Clears full compilation class in JAX by patrick-kidger
    # https://github.com/google/jax/issues/10828#issuecomment-1138428231
    process = psutil.Process()
    if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        obj.cache_clear()
        gc.collect()