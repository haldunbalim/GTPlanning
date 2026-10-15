import os
from setuptools import setup, find_packages
import unittest

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# string in below ...


def get_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(
    name="gtp",
    version="0.0.1",
    author="Anon Author",
    author_email="anon@gmail.com",
    description=("no description yet"),
    license="BSD",
    keywords="no keywords yet",
    packages=find_packages(),
    test_suite='setup.get_test_suite',
    classifiers=[

    ],
)
