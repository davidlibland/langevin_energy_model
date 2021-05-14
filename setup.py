from setuptools import setup
import dunamai

setup(version=dunamai.Version.from_git().serialize(dirty=True))
