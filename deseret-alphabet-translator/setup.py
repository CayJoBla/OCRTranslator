from setuptools import setup, find_packages

setup(
    name="deseret_alphabet_translator",
    version="0.1.0",
    description="Python package adapted from the project at https://www.2deseret.com/",
    packages=find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
)