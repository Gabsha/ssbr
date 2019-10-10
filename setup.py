import os
from setuptools import find_packages, setup

setup(
    name='ssbr',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    description="Self-supervised body part regression",
    url='http://gchartrand.com/posts/ssbr',
    author="Gabriel Chartrand",
)
