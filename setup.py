from setuptools import setup

def readme():
    with open('README.rst','r') as read_file:
        return(read_file.read())

setup(name='npmc',
        version='2.0',
        description='Python Wrapper for LAMMPS Monte Carlo simmulations of nanoparticle monolayers',
        url='https://github.com/smerz1989/np-mc',
        author='Steven Merz',
        license='MIT',
        packages=['npmc'],
        setup_requires=[
            'pytest-runner',
            ],
        tests_require=[
            'pytest',
            ],
        install_requires=[
            'matplotlib',
            'futures',
            'mock',
            'networkx',
            'numpy',
            'scipy',
            ],
        include_package_data=True,
        zip_safe=False)
