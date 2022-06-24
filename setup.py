import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="dirichlet-flows",
  version="0.0.1",
  author="Yasith Jayawardana",
  author_email="mail@yasith.dev",
  description="Normalizing Flows for Classification with Uncertainty",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/yasithdev/dirichlet-flows",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)