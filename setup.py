# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['SBART',
 'SBART.Base_Models',
 'SBART.Components',
 'SBART.Instruments',
 'SBART.Masks',
 'SBART.ModelParameters',
 'SBART.Quality_Control',
 'SBART.Samplers',
 'SBART.data_objects',
 'SBART.outside_tools',
 'SBART.rv_calculation',
 'SBART.rv_calculation.RV_Bayesian',
 'SBART.rv_calculation.rv_stepping',
 'SBART.template_creation',
 'SBART.template_creation.stellar_templates',
 'SBART.template_creation.telluric_templates',
 'SBART.utils',
 'SBART.utils.RV_utilities',
 'SBART.utils.concurrent_tools',
 'SBART.utils.cython_codes',
 'SBART.utils.cython_codes.cubic_interpolation',
 'SBART.utils.math_tools',
 'SBART.utils.parameter_validators',
 'SBART.utils.paths_tools',
 'SBART.utils.tapas_downloader',
 'SBART.utils.telluric_utilities']

package_data = \
{'': ['*'],
 'SBART': ['resources/*', 'resources/atmosphere_profiles/*'],
 'SBART.utils.cython_codes': ['matmul/second_term.pyx'],
 'SBART.utils.cython_codes.cubic_interpolation': ['inversion/inverter.pyx',
                                                  'partial_derivative/partial_derivative.pyx',
                                                  'second_derivative/second_derivative.pyx']}

install_requires = \
['Cython>=0.29.28,<0.30.0',
 'PyYAML>=6.0,<7.0',
 'astropy>=5.0.4,<6.0.0',
 'astroquery>=0.4.6,<0.5.0',
 'corner>=2.2.1,<3.0.0',
 'emcee>=3.1.1,<4.0.0',
 'eniric>=0.5.1,<0.6.0',
 'iCCF>=0.3.10,<0.4.0',
 'loguru>=0.6.0,<0.7.0',
 'matplotlib>=3.5.1,<4.0.0',
 'numpy>=1.22.3,<2.0.0',
 'python-dateutil>=2.8.2,<3.0.0',
 'requests>=2.27.1,<3.0.0',
 'tabletexifier>=0.1.9,<0.2.0',
 'tqdm>=4.64.0,<5.0.0']

setup_kwargs = {
    'name': 'sbart',
    'version': '0.1.0',
    'description': 'semi-Bayesian Radial Velocities',
    'long_description': None,
    'author': 'Kamuish',
    'author_email': 'andremiguel952@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
