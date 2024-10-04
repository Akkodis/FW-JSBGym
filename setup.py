from setuptools import setup, find_packages

setup(name='fw_jsbgym', 
      version='0.0.1',
      packages=find_packages(),
      package_data={'fw_jsbgym': ['trim/*.yaml', 'config/*.yaml', 'fdm_descriptions/',
                               'initial_conditions/', 'visualizers/']},
      )