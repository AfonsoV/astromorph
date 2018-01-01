import configparser
import os
import sys



# if __name__ == "__main__":
config = configparser.ConfigParser()
__folder__ = "astromorph" ## change to automatically set package folder
config.read("%s/astromorph.cfg"%(__folder__))
print("%s/astromorph.cfg"%(__folder__))

VERSION = [int(i) for i in config["package"]["version"].split(".")]
__version__ = "%i.%i.%i"%(VERSION[0],VERSION[1],VERSION[2])


from . import cosmology
from . import utils
from . import CAS
