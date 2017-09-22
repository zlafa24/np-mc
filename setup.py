from setuptools import setup

setup(name='npmc',
        version='2.0',
        description='Python Wrapper for LAMMPS Monte Carlo simmulations of nanoparticle monolayers',
        url='https://github.com/smerz1989/np-mc',
        author='Steven Merz',
        license='MIT',
        packages=['npmc'],
        install_requires=[
            'matplotlib',
            'futures',
            'lammps',
            'mock',
            'networkx',
            'numpy',
            'scipy',
            ],
        zip_safe=False)
