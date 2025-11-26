#!/usr/bin/env python3
"""
VeAgentBench CLI 安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="veagentbench",
    version="1.0.0",
    author="VeAgentBench Team",
    author_email="team@veagentbench.com",
    description="Agent评测工具 - VeAgentBench",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/veagentbench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "requests>=2.25.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "veagentbench=veagentbench.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
