from .spgl1 import *
from .lsqr import lsqr


try:
    from .version import version as __version__
except ImportError:
    __version__ = '0.0.0'