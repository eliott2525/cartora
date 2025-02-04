from setuptools import setup, find_packages

setup(
    name="antenna_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "geopy>=2.4.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for analyzing antenna locations and calculating distances",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 