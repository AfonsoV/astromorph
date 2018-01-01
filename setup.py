from distutils.core import setup


setup(
  name = 'astromorph',
  packages = ['astromorph'], # this must be the same as the name above
  version = '0.1.0',
  description = 'A python library for galaxy morphology',
  author = 'Bruno Ribeiro',
  author_email = 'brunorlr@gmail.com',
  url = 'https://github.com/afonsov/astromorph', # use the URL to the github repo
  download_url = 'https://github.com/afonsov/astromorph/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['astronomy', 'galaxies', 'morphology'], # arbitrary keywords
  classifiers = [],
  include_package_data=True
)
