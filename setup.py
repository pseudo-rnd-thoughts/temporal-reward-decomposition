"""Setups the project."""

from setuptools import setup

setup(
    name="TemporalRewardDecomposition",
    version="1.0.0",
    author="Mark Towers",
    author_email="mt5g17@soton.ac.uk",
    description="Implementation of 'Temporal Reward Decomposition'",
    license="MIT",
    keywords=["Reinforcement Learning", "Explanability AI"],
    python_requires=">=3.8",
    packages=["temporal_reward_decomposition"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
)
