import os
from distutils.core import setup
from setuptools import find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = """
        SPGL1: A solver for large-scale sparse reconstruction.
        """

# Setup
setup(
    name='spgl1',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='E. van den Berg, M. P. Friedlander (original MATLAB authors). David Relyea and contributors (python port).',
    author_email='drrelyea@gmail.com',
    url='https://github.com/drrelyea/SPGL1_python_port',
    install_requires=['numpy >= 1.15.0', 'scipy'],
    packages=find_packages(exclude=['pytests']),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('spgl1/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
