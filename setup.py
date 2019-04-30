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
          'spacy',
          'textacy',
          'flask',
          'keras', #requires backend installed such as tensorflow
          'GEOparse' #GEOquery inspired python package
      ],
      extras_require = {
          'documentation': ['Sphinx']
      },
      package_data = {
          'sina': [
              'templates/index.html',
              'static/js/main.js',
              'static/js/annotesto.js'
          ]
      },
      include_package_data = True,
      zip_safe = False,
      #entry_points = {
      #    'console_scripts': ['getLSDataset=LSD.command_line:main'],
      #},
      test_suite = 'nose.collector',
      tests_require = ['nose']
)

#To install with symlink, so that changes are immediately available:
#pip install -e . 
