from setuptools import setup, find_packages

setup(
    name='asc-analyzer',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
    "asc_analyzer": ["data/*.json"]
    },
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'asc-analyzer=asc_analyzer.cli:main',
        ],
    },
    description='asc-analyzer',
    author='Hakyung Sung',
    author_email='hksung001@gmail.com',
)
