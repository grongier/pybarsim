"""Setup for the pybarsim package

See:
https://github.com/pypa/sampleproject/blob/master/setup.py
"""

# LICENCE GOES HERE

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pybarsim',
    version='v0.0.1',
    description='Python package to simulate wave-dominated shallow-marine environments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.tudelft.nl/grongier/pybarsim',
    author='Guillaume Rongier',
    license='none',
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',        
    ],
    keywords='stratigraphic modeling, shallow-marine, event-based',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'scipy', 'numba', 'xarray', 'pyvista'],
    project_urls={  # Optional
        'Reference': 'https://doi.org/10.1016/S0025-3227(03)00144-0',
        'Source': 'https://gitlab.tudelft.nl/grongier/pybarsim',
    },
)
