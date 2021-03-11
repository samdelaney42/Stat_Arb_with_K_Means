import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
	name = "Stat_Arb_with_K_Means_samdelaney42",
	version = "0.0.1",
	author = "Sam Delaney", 
	author_email = "smd575@nyu.edu",
    description = "Cluster stocks to find pairs and backtest them in stat arb model",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/samdelaney42/Stat_Arb_with_K_Means",
    project_urls = {
        "Bug Tracker": "https://github.com/samdelaney42/Stat_Arb_with_K_Means/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src"),
    python_requires = ">=3.6",
)