import pathlib
from setuptools import setup, find_packages

# CWD = pathlib.Path(__file__).absolute().parent

setup(name='jsbgym', 
      version='0.0.1',
      packages=find_packages(),
      package_data={'jsbgym': ['trim/*.yaml', 'config/*.yaml', 'fdm_descriptions/',
                               'initial_conditions/', 'visualizers/']},
      )