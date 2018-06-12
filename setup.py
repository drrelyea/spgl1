from distutils.core import setup

setup(
    name='spgl1',
    description='SPGL1: A solver for large-scale sparse reconstruction',
    long_description='SPGL1 is a Matlab solver for large-scale one-norm regularized least squares',
    author='E. van den Berg, M. P. Friedlander (original MATLAB authors).  David Relyea and contributors (python port).',
    author_email='drrelyea@gmail.com',
    url='https://github.com/drrelyea/SPGL1_python_port',
    license='Lesser GPL 2.1',
    install_requires=[
        'setuptools',
        'numpy',
        ],
    packages=['spgl1'],
    )
