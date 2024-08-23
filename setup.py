import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="utils_jgm",
    version="0.7.0",
    author="J.G. Makin",
    author_email="jgmakin@gmail.com",
    description="general python utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgmakin/utils_jgm",
    packages=setuptools.find_packages(),
    package_data={
        '': ['utils_config.json']
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'matplotlib', # ==3.6 because of incompat with tikzplotlib
        'scipy',
        # 'samplerate', causes problems on some machines....
        'ipywidgets', 'tikzplotlib',
        # 'bqplot'
    ],
)
