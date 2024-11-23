from setuptools import setup, find_packages

setup(
    name="docdost",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "redis>=4.5.0",
        "redisvl>=0.1.0",
        "spacy>=3.5.0",
        "PyMuPDF>=1.22.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    author="Namra Maniar",
    author_email="namra4122@gmail.com",
    description="Local Document Dost for you",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)