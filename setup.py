import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="deeptabular",
    version="0.0.2",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="Deep neural networks for tabular data",
    license="APACHE",
    keywords="tabular",
    url="https://github.com/CVxTz/DeepTabular",
    packages=["deeptabular"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
