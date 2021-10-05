from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


def requirements():
    with open("requirements.txt") as f:
        ret = [l.strip() for l in f.readlines()]
    return ret


setup(
    name="roveranalyzer",
    version="1.5.10",
    description="roVer results analysis tool",
    long_description=readme(),
    author="Stefan Schuhbäck",
    author_email="stefan.schuhbaeck@hm.edu",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    # scripts=[''],
    install_requires=requirements(),
    zip_safe=False,
)
