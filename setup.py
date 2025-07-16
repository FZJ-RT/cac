from setuptools import setup

setup(
    name='cac',
    version='2.0',
    description='A library for analyzing chip properties',
    packages=['cac', 'cac.model','cac.trainer','cac.utilities'],
    package_dir={'': 'src'},
)
