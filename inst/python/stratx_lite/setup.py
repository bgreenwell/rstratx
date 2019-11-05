from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))


setup(
    name='stratx_lite',
    version='0.1',
    description='A bare-bones (and reticulate-friendly) version of the stratx Python library by Terence Parr',
    url='http://github.com/bgreenwell/stratx',
    author='Brandon Greenwell (originally developed by Terence Parr)',
    author_email='greenwell.brandon@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['sklearn','pandas','numpy','scipy']
)
