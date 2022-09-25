from setuptools import setup, find_packages

setup(
    name='ddrr', 
    version='1.0', 
    description="Differentiable digital reconstructed radiograpy (DRR)",
    author='Lin Tian',
    author_email='lintian@cs.unc.edu',
    package_dir={"":"src"},
    packages=find_packages(where='src'),
    install_requires=[
        "torch",
        "numpy"
    ],
    )