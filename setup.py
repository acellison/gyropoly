
from setuptools import setup

setup(
    packages=['gyropoly'],
    version='0.1'
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'sympy',
        'mpmath',
        'pathos',
        'dedalus_sphere',
        'scikit-umfpack',
    ]
)
