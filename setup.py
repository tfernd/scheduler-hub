from setuptools import setup, find_packages

setup(
    name="scheduler_hub",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # TODO add
        "torch",
        "einops",
    ],
    author="Thales Fernandes",
    author_email="thalesfdfernandes@gmail.com",
    # description="",
    url="https://github.com/tfernd/scheduler-hub",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
