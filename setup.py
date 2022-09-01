import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qpm-amrb",
    version="0.0.1",
    author="Yasith Jayawardana",
    author_email="mail@yasith.dev",
    description="Explainable Image Classification of Antimicrobial Drug Resistant Bacteria Cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasithdev/qpm-amrb-new",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
