import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myProCon",
    version="0.0.1",
    author="Edgar Qian",
    author_email="qianneng_se@163.com",
    description="ProCon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loststarsss/ProCon_python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
