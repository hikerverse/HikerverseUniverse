from setuptools import setup, find_packages

setup(
    name="hikerserveUniverse",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A spacecraft simulation project with power storage components.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hikerservespacecraft",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here, e.g., "numpy>=1.21.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            # Add command-line scripts here, e.g., "hikerservespacecraft=hikerservespacecraft.cli:main"
        ],
    },
    include_package_data=True,
)