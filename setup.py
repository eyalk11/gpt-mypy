from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
import pkg_resources
import setuptools
import pathlib
desc = 'Python minor issue resolver using gpt by openai!'
long_desc = 'Solves linter errors using guidance and GPT api. It runs mypy(for now) on targeted file and then uses gpt to try to fix the issues (espcially good for minor nagging issues). Displays a diff file for the required changes and ask you if you want to apply.'
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='gpt-linter',
    version='0.1.0',
    packages=['gpt_linter'],
    url='https://github.com/eyalk11/mypy-gpt',
    license=' AGPL-3.0 license',
    author='ekarni',
    author_email='eyalk5@gmail.com',
    description=desc,
    long_description=long_desc,
    install_requires=install_requires
)
