from setuptools import setup

setup(
    name="cmtpy",
    packages=["cmtpy"],
    version="0.0.1",
    description="Tools for manipulating image contrast",
    author="vs",
    author_email="vikrantshah+py@gmail.com",
    license='MIT',
    url="https://github.com/vik748/cmtpy",
    keywords=["contrast", "histogram", "enhancement"],
    classifiers=[],
    install_requires=[
        'scipy',
        'numpy'
    ],
)
