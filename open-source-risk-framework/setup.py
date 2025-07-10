from setuptools import setup, find_packages

setup(
    name="open-source-risk-framework",
    version="0.1.0",
    description="Modular open-source risk framework for credit, fraud, operational, market, and regulatory risk.",
    author="Souptik Chakraborty",
    author_email="souptikc80@gmail.com",
    url="https://github.com/Souptik96/open-source_risk_framework",
    packages=find_packages(include=["risk_framework", "risk_framework.*"]),
    include_package_data=True,
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.6.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "networkx>=2.8.0",
        "seaborn>=0.11.2",
        "jupyter",
        "streamlit",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "risk-framework=risk_framework.__main__:main"
        ]
    },
)
