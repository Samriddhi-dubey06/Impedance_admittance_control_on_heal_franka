#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='ds_control',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[],  # add dependencies here if needed
)