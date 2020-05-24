from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="cmtpy",
    packages=["cmtpy"],
    version="0.0.3",
    description="Tools for manipulating image contrast",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="vs",
    author_email="vikrantshah+py@gmail.com",
    license='MIT',
    url="https://github.com/vik748/cmtpy",
    keywords=["contrast", "histogram", "enhancement"],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research',      # Define that your audience are developers
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'scipy',
        'numpy'
    ],
)