import codecs
from setuptools import find_packages
from setuptools import setup

install_requires = [
    'gym==0.22.0',
    "torch>=1.11.0",
    "numpy>=1.23.4",
    "filelock>=3.8.0",
    "Pillow>=9.2.0",
    "optuna>=3.0.3",
    "pandas>=1.5.0",
    "matplotlib>=3.6.1",
    "pygame>=2.1.0",
    "quik-config>=1.7.0",
    "box2d-py>=2.3.5",
    "blissful-basics>=0.2.18",
    "super-map>=1.3.1",
    "super-hash>=1.2.2",
    'mujoco-py<2.2,>=2.1',
    'plotly>=5.11.0',
]

test_requires = [
    'pytest',
    'attrs<19.2.0',  # pytest does not run with attrs==19.2.0 (https://github.com/pytest-dev/pytest/issues/3280)  # NOQA
]

setup(name='pfrl',
      version='0.3.0',
      description='PFRL, a deep reinforcement learning library',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Yasuhiro Fujita',
      author_email='fujita@preferred.jp',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires)
