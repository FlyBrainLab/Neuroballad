from setuptools import setup
from setuptools import find_packages

install_requires=[
      'networkx',
      'numpy',
      'h5py',
      'matplotlib',
      # 'neurokernel'
]

setup(name='Neuroballad',
      version='0.1.0',
      description='Neural Circuit Simulation for Python',
      author='Mehmet Kerem Turkcan',
      author_email='mkt2126@columbia.edu',
      url='',
      install_requires=install_requires,
      download_url='',
      license='BSD-3-Clause',
      packages=find_packages())