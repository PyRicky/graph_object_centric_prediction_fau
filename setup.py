
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='gocp',
    version='1.0.0',
    description='',
    long_description=readme,
    author='',
    author_email='',
    url='https://github.com/fau-is/gocp',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

