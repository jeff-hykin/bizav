import codecs
from setuptools import find_packages
from setuptools import setup

install_requires = [
    'gym==0.22.0',
    'pillow',
    'pygame',
    "torch>=1.12.1",
    "numpy>=1.23.4",
    "filelock>=3.8.0",
    "Pillow>=9.2.0",
    "optuna>=3.0.3",
    "pandas>=1.5.0",
    "matplotlib>=3.6.1",
    "pygame2.1.0",
    "quik-config>=1.6.0",
    "box2d-py2.3.5",
    "blissful-basics0.2.1",
    "super-map1.3.1",
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
