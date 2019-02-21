from setuptools import setup, find_packages

setup(name = 'sina',
      version = '0.0.1',
      description = 'Search Indexed Nomenclature Associations',
      url = 'https://github.com/dicaso/sina',
      author = 'Christophe Van Neste',
      author_email = 'christophe.vanneste@kaust.edu.sa',
      license = 'MIT',
      packages = find_packages(),
      python_requires='>3.6',
      install_requires = [
          'requests',
          'pandas',
          'bidali',
          'pymongo',
          'whoosh',
          'spacy'
      ],
      extras_require = {
          'documentation': ['Sphinx']
      },
      zip_safe = False,
      #entry_points = {
      #    'console_scripts': ['getLSDataset=LSD.command_line:main'],
      #},
      test_suite = 'nose.collector',
      tests_require = ['nose']
)

#To install with symlink, so that changes are immediately available:
#pip install -e . 
