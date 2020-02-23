from setuptools import setup
from setuptools import find_packages


setup(name='Neuroballad',
      version='0.1.0',
      description='Neural Circuit Simulation for Python',
      author='Mehmet Kerem Turkcan',
      author_email='mkt2126@columbia.edu',
      url='',
      download_url='',
      license='BSD-3-Clause',
      packages=find_packages(),
      install_requires=[
          "networkx<=2.3",
          "pygraphviz",
          "tqdm",
          "pycuda",
          "numpy",
          "matplotlib",
          # "neurokernel"
      ])
