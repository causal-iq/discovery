from setuptools import setup, find_packages

setup(
    name="causaliq",
    version="0.0.2",
    description="Placeholder for the causaliq package.",
    url="https://github.com/causal-iq/discovery",
    author="Ken Kitson",
    author_email="kenkitson@gmail.com",
    packages=find_packages(include=["causaliq", "causaliq.*"]),
    python_requires=">=3.7",
)