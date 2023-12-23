# pyBarSim

[![DOI:10.4121/60f148ee-a793-4d16-b1b6-c03442403db1.v1](http://img.shields.io/badge/DOI-10.4121/60f148ee--a793--4d16--b1b6--c03442403db1.v1-B31B1B.svg)](https://doi.org/10.4121/60f148ee-a793-4d16-b1b6-c03442403db1.v1) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/grongier/pybarsim/master?filepath=examples)

pyBarSim is a Python package to simulate wave-dominated shallow-marine environments using [Storms (2003)](https://doi.org/10.1016/S0025-3227(03)00144-0)'s BarSim.

![](https://raw.githubusercontent.com/grongier/pybarsim/master/image.jpg)

## Installation

You can directly install pyBarSim from pip:

    pip install pybarsim

Or from GitHub using pip:

    pip install git+https://github.com/grongier/pybarsim.git

## Usage

Basic use:

```
import numpy as np
from pybarsim import BarSim2D
import matplotlib.pyplot as plt

# Set the parameters
run_time = 10000.
barsim = BarSim2D(np.linspace(1000., 900., 200),
                  np.array([(0., 950.), (run_time, 998.)]),
                  np.array([(0., 25.), (run_time, 5.)]),
                  spacing=100.)
# Run the simulation
barsim.run(run_time=10000., dt_fair_weather=15., dt_storm=1.)
# Interpolate the outputs into a regular grid
barsim.regrid(900., 1000., 0.5)
# Compute the mean grain size
barsim.finalize(on='record')
# Plot the median grid size in the regular grid
barsim.record_['Mean grain size'].plot(figsize=(12, 4))
plt.show()
```

For a more complete example, see the Jupyter notebook [using_pybarsim.ipynb](examples/using_pybarsim.ipynb) or the Binder link above.

## Citation

If you use pyBarSim in your research, please cite the original article:

> Storms, J.E.A. (2003). Event-based stratigraphic simulation of wave-dominated shallow-marine environments. *Marine Geology*, 199(1), 83-100. doi:10.1016/S0025-3227(03)00144-0

Here is the corresponding BibTex entry if you use LaTex:

	@Article{Storms2003,
		author="Storms, Joep E.A.",
		title="Event-based stratigraphic simulation of wave-dominated shallow-marine environments",
		journal="Marine Geology",
		year="2003",
		volume="199",
		number="1",
		pages="83--100",
		issn="0025-3227",
		doi="https://doi.org/10.1016/S0025-3227(03)00144-0",
	}

## Credits

This software was written by:

| [Guillaume Rongier](https://github.com/grongier) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-5910-6868)</br> | [Joep Storms](https://www.tudelft.nl/en/ceg/about-faculty/departments/geoscience-engineering/sections/applied-geology/staff/academic-staff/storms-jea) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-8902-8493)</br> | [Andrea Cuesta Cano](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-engineering/sections/applied-geology/staff/phd-students/cuesta-cano-a) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-7017-6031)</br> |
| :---: | :---: | :---: |

## License

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program pyBarSim written by the Author(s). Prof. Dr. Ir. J.D. Jansen, Dean of the Faculty of Civil Engineering and Geosciences

&#169; 2023, G. Rongier, J.E.A. Storms, A. Cuesta Cano

This work is licensed under a MIT OSS licence.
