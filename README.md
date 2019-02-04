# Search Indexed Nomenclature Associations

Text mining support package. Developed at KAUST by Christophe Van Neste,
in collaboration with Adil Salhi. Under guidance from Vladimir Bajic.

Packaged named in reference to Ibn Sina, who in his time mastered all
known scientific knowledge, without the help of a computer.

## Requirements

- Install python3 from https://www.python.org/downloads/ (version 3.6.2)
- Install git from https://git-scm.com/downloads
- https://virtualenvwrapper.readthedocs.io/en/latest/

## User installation

    pip install git+https://gitlab.kaust.edu.sa/vannescm/sina.git


## Installation for developers

Open `Terminal` and copy paste below line by line:

     mkdir -p ~/{repos,LSData/cache} && cd ~/repos
     git clone https://gitlab.kaust.edu.sa/vannescm/sina.git
     SINADIR=~/repos/sina
     mkvirtualenv -a $SINADIR -i ipython -i Sphinx \
                  -r $SINADIR/requirements.txt sina
     python setup.py test # runs all the tests
     deactivate