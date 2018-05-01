
from setuptools import setup, find_packages

setup(name = "lshlink",
      version = "1.1.0",
      author='Walker Harrison, Lisa Lebovici',
      author_email='walker.harrison@duke.edu, lisa.lebovici@duke.edu',
      description='An implementation of locality-sensitive hashing for agglomerative hierarchical clustering',
      url='https://github.com/lisalebovici/LSHLinkClustering',
      py_modules = ['lshlink'],
      packages = find_packages(),
      scripts=['run_lshlink.py'],
      license = 'MIT License',
      python_requires='>=3'
      )