from .config import ConfigFile,__folder__
from . import cosmology
from . import utils
from . import plot_utils
from . import CAS
from . import ADD
from . import gm20
from . import MID
from . import TMC
from . import simulation
from . import galfit
from . import lensing

__version__ = ConfigFile["package"]["version"]
VERSION = [int(i) for i in __version__.split(".")]
