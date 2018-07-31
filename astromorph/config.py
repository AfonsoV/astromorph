# Licensed under GNU GENERAL PUBLIC LICENSE Version 3 - see LICENSE.txt
"""
Utility module to load default configuration file for the astromorph package
"""

import configparser

ConfigFile = configparser.ConfigParser()
__folder__ = __file__[:-10]
ConfigFile.read("%s/astromorph.cfg"%(__folder__))
