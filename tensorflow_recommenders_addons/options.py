import os
import platform
import warnings
import traceback

try:
  TF_RECOMMENDERS_ADDONS_PY_OPS = bool(
      int(os.environ["TF_RECOMMENDERS_ADDONS_PY_OPS"]))
except KeyError:
  if platform.system() == "Linux":
    TF_RECOMMENDERS_ADDONS_PY_OPS = False
  else:
    TF_RECOMMENDERS_ADDONS_PY_OPS = True

FALLBACK_WARNING_TEMPLATE = """{}

The {} C++/CUDA custom op could not be loaded.
For this reason, Recommenders Addons will fallback to an implementation written
in Python with public TensorFlow ops. There worst you might experience with
this is a moderate slowdown on GPU. There can be multiple
reason for this loading error, one of them may be an ABI incompatibility between
the TensorFlow installed on your system and the TensorFlow used to compile
TensorFlow Recommenders Addons' custom ops. The stacktrace generated
when loading the shared object file was displayed above.

If you want this warning to disappear, either make sure the TensorFlow installed
is compatible with this version of Recommenders Addons, or tell TensorFlow
Recommenders Addons to prefer using Python implementations and not custom
C++/CUDA ones. You can do that by changing the TF_RECOMMENDERS_ADDONS_PY_OPS
flag either with the environment variable:
```bash
TF_RECOMMENDERS_ADDONS_PY_OPS=1 python my_script.py
```
or in your code, after your imports:
```python
import tensorflow_recommenders_addons as tfra
import ...
import ...

tfra.options.TF_RECOMMENDERS_ADDONS_PY_OPS = True
```
"""


def warn_fallback(op_name):
  warning_msg = FALLBACK_WARNING_TEMPLATE.format(traceback.format_exc(),
                                                 op_name)
  warnings.warn(warning_msg, RuntimeWarning)
  global TF_RECOMMENDERS_ADDONS_PY_OPS
  TF_RECOMMENDERS_ADDONS_PY_OPS = True
