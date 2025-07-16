from setuptools import setup

setup(
    name='cac',
    version='1.0',
    description='A library for analyzing chip properties',
    packages=['cac', 'cac.model','cac.trainer','cac.utilities'],
    package_dir={'': 'src'},
)
