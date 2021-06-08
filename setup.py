#!/usr/bin/env python

# Always prefer setuptools over distutils
import os
from setuptools import setup, find_packages

# Grab from README file: long_description
with open("README.md", "r") as f:
    long_description = f.read()

def do_setup():
    setup(
        name="connectedComponents",
        version=0.1,
        description="DEM Connected Component calculation for debris-flow detection",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/UP-RS-ESP/DEM_ConnectedComponents",
        author="Ariane Mueting",
        author_email="mueting@uni-potsdam.de",

        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        download_url="https://github.com/UP-RS-ESP/DEM_ConnectedComponents",
        keywords="DEM, ConnectedComponents, river networks, debris flow, drainage area, natural hazard, geology, geomorphology, high resolution DEM, remote sensing",

        # package discovery
        packages=find_packages(include=['src', 'ConnectedComponents.*']),
        install_requires=[
        'pandas',
        'numpy',
        'rasterio'
        ],

        # dependencies
        python_requires=">=3.6",
        project_urls={
            "Bug Reports": "https://github.com/UP-RS-ESP/DEM_ConnectedComponents/issues",
            "Documentation": "https://github.com/UP-RS-ESP/DEM_ConnectedComponents/examples",
            "Source": "https://github.com/UP-RS-ESP/DEM_ConnectedComponents/",
        },
    )


if __name__ == "__main__":
    do_setup()
