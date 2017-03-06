# -*- coding: UTF-8 -*-
from setuptools import setup, find_packages


def slurp(filename):
    with open(filename) as text_file:
        return text_file.read()


setup(
    name='sensibility',
    version='0.2.dev0',

    description='Syntax error detecter and fixer',
    long_description=slurp('README.md'),
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
    install_requires=[
        'Keras>=1.2.2',
        'numpy>=1.11.0',
        'h5py>=2.6.0',
        'tqdm',
        'blessings',
        'more-itertools>=2.3',
    ],
    extra_require={
        'test': ['py.test', 'mypy', 'pystyleguide']
    },
)
