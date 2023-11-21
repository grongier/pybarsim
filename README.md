# pyBarSim

pyBarSim is a Python package to simulate wave-dominated shallow-marine environments using [Storms (2003)](https://doi.org/10.1016/S0025-3227(03)00144-0)'s BarSim.

## Installation

You can directly install pyBarSim from GitLab using pip:

    pip install git+https://gitlab.tudelft.nl/grongier/pybarsim.git

## Use

Basic use:

```
import numpy as np
from pybarsim import BarSim2D

# Set the parameters
duration = 10000.
barsim = BarSim2D(np.linspace(1000., 900., 200),
                  np.array([(0., 950.), (duration, 998.)]),
                  np.array([(0., 25.), (duration, 5.)]),
                  spacing=100.)
# Run the simulation
barsim.run(duration=10000., dt_min=1., dt_fw=15.)
# Interpolate the outputs into a regular grid
barsim.regrid(900., 1000., 0.5)
# Compute the mean grain size
barsim.summarize()
# Plot the median grid size in the regular grid
barsim.record_['Mean grain size'].plot(figsize=(12, 4))
```

For a more complete example, see the Jupyter notebook [using_pybarsim.ipynb](examples/using_pybarsim.ipynb).

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

## Authors

* Guillaume Rongier ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0002-5910-6868](https://orcid.org/0000-0002-5910-6868)

* Joep Storms ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0002-8902-8493](https://orcid.org/0000-0002-8902-8493)

* Andrea Cuesta Cano ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0002-7017-6031](https://orcid.org/0000-0002-7017-6031)

## License

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program pyBarSim written by the Author(s). Prof. Dr. Ir. J.D. Jansen, Dean of the Faculty of Civil Engineering and Geosciences

&#169; 2023, G. Rongier, J.E.A. Storms, A. Cuesta Cano

This work is licensed under a MIT OSS licence.
