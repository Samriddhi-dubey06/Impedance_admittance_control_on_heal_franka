#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='snap_detector',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[],  
)
