from setuptools import setup, find_namespace_packages

setup(
    name="posedreamer",
    version="0.1.0",
    packages=find_namespace_packages(include=["posedreamer", "posedreamer.*"]),
    install_requires=[],
)
