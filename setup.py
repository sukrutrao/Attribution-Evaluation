from setuptools import setup, find_packages

setup(name='attribution_evaluation',
      version='1.0',
      packages=find_packages(),
      install_requires=["captum == 0.4.1",
                        "matplotlib == 3.5.1",
                        "numpy == 1.21.2",
                        "pandas == 1.3.5",
                        "scikit_image == 0.19.1",
                        "seaborn == 0.11.2",
                        "torch == 1.10.1",
                        "torchvision == 0.11.2",
                        "tqdm == 4.62.3"])
