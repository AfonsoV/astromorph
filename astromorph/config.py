import configparser

ConfigFile = configparser.ConfigParser()
__folder__ = __file__[:-10]
ConfigFile.read("%s/astromorph.cfg"%(__folder__))
