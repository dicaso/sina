from setuptools import setup, find_packages

setup(name='sina',
      version='0.0.1',
      description='Search Indexed Nomenclature Associations',
      url='https://github.com/dicaso/sina',
      author='Christophe Van Neste',
      author_email='christophe.vanneste@kaust.edu.sa',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'argetype',  # for settings CLI
          'appdirs',  # cache/data dirs for app
          'dill',     # improved pickle
          'plumbum',  # subcommands
          'kindi',
          'requests',
          'pandas',
          'bidali',
          'pymongo',
          'whoosh',
          'flask',
          'scikit-learn',
          'scikit-multilearn',
          'imbalanced-learn',
          'spacy',
          'scispacy',
          'textacy',
          'gensim',
          'tensorflow',
          'GEOparse'  # GEOquery inspired python package
      ],
      extras_require={
          'documentation': ['Sphinx'],
          'nbpaper': ['node2vec', 'networkx'],
      },
      package_data={
          'sina': [
              'templates/index.html',
              'static/js/main.js',
              'static/js/annotesto.js'
          ]
      },
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'sina_nb=sina.paperwork.neuroblastoma:main'
          ],
      },
      test_suite='nose.collector',
      tests_require=['nose']
      )

# To install with symlink, so that changes are immediately available:
# pip install -e .
