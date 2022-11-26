import os
from setuptools import setup, find_packages


setup(
    name='ctapipe_io_nectarcam',
    packages=find_packages(exclude=["ctapipe_io_nectarcam._dev_version"]),
    use_scm_version={"write_to": os.path.join("ctapipe_io_nectarcam", "_version.py")},
    description='ctapipe plugin for reading NectarCam files',
    install_requires=[
        'astropy~=4.2',
        'ctapipe~=0.12',
        'numpy~=1.22.4',
        'protozfits~=2.0',
        'setuptools_scm',
    ],
    author='Franca Cassol',
    author_email='cassol@cppm.in2p3.fr',
    license='MIT',
)
