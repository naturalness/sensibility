# -*- coding: UTF-8 -*-
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

def slurp(filename):
    with open(filename, encoding='UTF-8') as text_file:
        return text_file.read()


setup(
    name='sensibility',
    version='0.2.dev0',

    description='Syntax error detecter and fixer',
    long_description=slurp(path.join(here, 'README.rst')),
    url='https://github.com/eddieantonio/training-grammar-guru',

    author='Eddie Antonio Santos',
    author_email='easantos@ualberta.ca',

    license='Apache',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',
        'Environment :: Console',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3.6',
    ],

    packages=find_packages(),

    # Node.JS app is NOT zip safe!
    zip_safe=False,

    install_requires=[
        # TODO: upgrade to 2.x!
        'Keras>=1.2.2',
        'SQLAlchemy>=1.1.9',
        'blessings',
        'github3.py==0.9.6',
        'h5py>=2.6.0',
        'lazy-object-proxy>=1.3.1',
        'more-itertools>=2.3',
        'numpy>=1.11.0',
        'numpy>=1.11.0',
        'python-dateutil>=2.6.0',
        'redis>=2.10.5',
        'requests==2.13.0',
        'tqdm',
    ],
    extras_require={
        'test': [
            'pytest',
            'mypy',
            'pystyleguide'
        ]
    },
)
