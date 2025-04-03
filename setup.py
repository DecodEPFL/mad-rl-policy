from setuptools import setup, find_packages

setup(
    name="mad-rl-policy",
    version="0.1.0",
    author="Sucheth Shenoy",
    author_email="sucheth.shenoy@epfl.ch",
    description="MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DecodEPFL/mad-rl-policy",
    packages=find_packages(),
    install_requires=[
        "torch==2.3.1",
        "numpy==2.0.0",
        "gymnasium==0.29.1",
        "matplotlib==3.9.0",
        "pandas==2.2.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="CC-BY-4.0",
)
