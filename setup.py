"""Setup for the pyBarSim package

See:
https://github.com/pypa/sampleproject/blob/master/setup.py
"""

# MIT License

# Copyright (c) 2023 Guillaume Rongier, Joep Storms, Andrea Cuesta Cano

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright notice: Technische Universiteit Delft hereby disclaims all copyright
# interest in the program pyBarSim written by the Author(s).
# Prof. Dr. Ir. J.D. Jansen, Dean of the Faculty of Civil Engineering and Geosciences


# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path


# Get the long description from the README file
with open('README.md') as f:
    long_description = f.read()

setup(
    name='pyBarSim',
    version='v0.0.1',
    description='Python package to simulate wave-dominated shallow-marine environments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/grongier/pybarsim',
    author='Guillaume Rongier',
    license='MIT',
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
