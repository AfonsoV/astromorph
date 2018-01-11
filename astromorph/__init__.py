from .config import ConfigFile,__folder__
from . import cosmology
from . import utils
from . import CAS
from . import simulation

__version__ = ConfigFile["package"]["version"]
VERSION = [int(i) for i in __version__.split(".")]
