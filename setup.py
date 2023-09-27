from setuptools import find_packages, setup

install_requires = [
    "category-encoders>=2.6.1",
    "gretel-client>=0.16.11",
    "optuna>=3.2.0",
    "pandas>=1.5.3",
    "scikit-learn>=1.3.0",
    "sdmetrics>=0.10.1",
    "smart-open>=5.2.1",
    "tabulate>=0.8.9",
    "tqdm>=4.65.0",
    "xgboost>=2.0.0",
]

setup(
    name="gretel_tuner",
    version="0.0.1",
    python_requires=">=3.9",
    packages=find_packages("src"),
    package_dir={"": "src"},
    license="https://gretel.ai/license/source-available-license",
    install_requires=install_requires,
)
