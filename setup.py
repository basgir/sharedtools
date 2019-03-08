import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='sharedtools',
    version='0.136',
    author="SB Ca",
    author_email="canada@swissborg.com",
    description="Shared tools for jupyter notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swissborgcanada/sharedtools",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
