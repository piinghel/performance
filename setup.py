from struct import pack
from setuptools import setup, find_packages

setup(
    name="performance_evaluation",
    version="0.1.0",
    packages=find_packages("performance_evaluation"),
    install_requires=["pandas>=1.3.4" "scipy>=1.7.2", "numpy>=1.19.2"],
)