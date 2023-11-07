from setuptools import setup, find_packages

setup(
    name='asset_hub',
    version='0.1.3',
    packages=find_packages(include=['asset_hub', 'asset_hub.*']),
    install_requires=[
        'certifi==2023.7.22',
        'charset-normalizer==3.2.0',
        'colorama==0.4.6',
        'idna==3.4',
        'requests==2.31.0',
        'tqdm==4.66.1',
        'urllib3==1.26.6'
    ]
)
