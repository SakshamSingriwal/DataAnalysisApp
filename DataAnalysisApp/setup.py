"""
Setup configuration for DataAnalysisApp package

This file enables installation as a Python package using:
    pip install -e .
"""

from setuptools import setup, find_packages

with open(\"README.md\", \"r\", encoding=\"utf-8\") as fh:
    long_description = fh.read()

with open(\"requirements.txt\", \"r\", encoding=\"utf-8\") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith(\"#\")]

setup(
    name=\"DataAnalysisApp\",
    version=\"1.0.0\",
    author=\"Saksham Singriwal\",
    author_email=\"contact@example.com\",
    description=\"Professional Data Analytics & Machine Learning Platform\",
    long_description=long_description,
    long_description_content_type=\"text/markdown\",\n    url=\"https://github.com/SakshamSingriwal/DataAnalysisApp\",\n    packages=find_packages(),\n    classifiers=[\n        \"Development Status :: 4 - Beta\",\n        \"Intended Audience :: Data Scientists\",\n        \"Topic :: Scientific/Engineering :: Artificial Intelligence\",\n        \"License :: OSI Approved :: MIT License\",\n        \"Programming Language :: Python :: 3\",\n        \"Programming Language :: Python :: 3.8\",\n        \"Programming Language :: Python :: 3.9\",\n        \"Programming Language :: Python :: 3.10\",\n        \"Programming Language :: Python :: 3.11\",\n    ],\n    python_requires=\">=3.8\",\n    install_requires=requirements,\n    include_package_data=True,\n    zip_safe=False,\n)\n"