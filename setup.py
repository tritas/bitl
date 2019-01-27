# -*- coding: utf-8 -*-
# Author: Aris Tritas
# Licence: BSD 3 clause

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def setup_package():
    setup(name='bitl',
          version='0.0.1',
          description='A package to benchmark multi-armed bandit algorithms',
          url='https://github.com/tritas/bitl',
          author='Aris Tritas',
          author_email='a.tritas@gmail.com',
          license='BSD',
          packages=['bitl'])


if __name__ == '__main__':
    setup_package()
