from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


def version():
    with open("crownetutils/version.txt") as f:
        return f.readline().strip()


def requirements():
    with open("requirements.txt") as f:
        ret = [l.strip() for l in f.readlines()]
    return ret


setup(
    name="crownetutils",
    version=version(),
    description="Collection of CrowNet simulation and analysis tool",
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
