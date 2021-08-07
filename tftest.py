import sys
print("sys path:", sys.path)

try:
    import tensorflow as tf
    # print(format(tf.__version__))
    print("tf version:", tf.version.VERSION, ", tf path:", tf.sysconfig.get_lib())
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.message)

try:
    import numpy as np
    print("numpy version:", np.version.version, ", numpy path:", np.get_include())
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.message)

try:
    import torch
    print("GPU is available for torch?", torch.cuda.is_available())
    print("torch version:", torch. __version__, ", torch path:", torch.__path__)
except ImportError as error:
    print("PyTorch is only necessary for evaluations using LSiM. Ignore torch ImportError otherwise.")
    print(error.__class__.__name__ + ": " + error.message)

print("Done")