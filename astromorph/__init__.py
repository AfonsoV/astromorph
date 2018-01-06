from .config import ConfigFile,__folder__
from . import cosmology
from . import utils
from . import CAS


VERSION = [int(i) for i in ConfigFile["package"]["version"].split(".")]
__version__ = "%i.%i.%i"%(VERSION[0],VERSION[1],VERSION[2])
