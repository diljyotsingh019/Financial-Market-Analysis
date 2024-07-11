from setuptools import setup, find_packages    
  
__version__ = "0.0.1"
REPO_NAME = ""
PKG_NAME= "financial-agent"
AUTHOR_USER_NAME = "diljyotsingh019"
AUTHOR_EMAIL = "diljyotsingh019@gmail.com"

setup(
    name=PKG_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A python package for Financial market analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    )