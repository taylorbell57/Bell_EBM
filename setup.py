import setuptools
from Bell_EBM import __version__, name

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = name,
    version = __version__,
    author = "Taylor James Bell",
    author_email = "taylor.bell@mail.mcgill.ca",
    description = "An object-oriented Energy Balance Model that can be used to model exoplanet observations.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/taylorbell57/Bell_EBM",
    license="MIT",
    package_data={"": ["LICENSE"]},
    packages = setuptools.find_packages(),
    classifiers = (
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data = True,
    zip_safe = True,
    install_requires = [
        "numpy", "scipy", "astropy", "matplotlib"]
)

