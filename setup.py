from struct import pack
from setuptools import setup, find_packages

setup(
    name="performance_analysis",
    version="0.0.1",
    packages=find_packages("performance_analysis"),
    install_requires=["pandas>=1.3.4" "scipy>=1.7.2", "numpy>=1.19.2"],
)