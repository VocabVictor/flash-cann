"""
Flash-CANN: Flash-Attention for Huawei Ascend NPU

A drop-in replacement for flash-attn using CANN backend.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_file = os.path.join("flash_attn", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def get_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="flash-cann",
    version=get_version(),
    author="Zhongheng Wu",
    author_email="wuzhongheng@smail.nju.edu.cn",
    description="Flash-Attention implementation for Huawei Ascend NPU using CANN",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/VocabVictor/flash-cann",
    project_urls={
        "Bug Tracker": "https://github.com/VocabVictor/flash-cann/issues",
        "Source Code": "https://github.com/VocabVictor/flash-cann",
        "Documentation": "https://github.com/VocabVictor/flash-cann#readme",
    },
    packages=find_packages(exclude=["tests", "benchmarks", "examples"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "pandas>=1.5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
