import os
import subprocess, sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info

class SetupAltoDevelop(develop):
    def run(self):
        develop.run(self)
        os.system("echo 'Setting up Alto parser'")
        runScriptInstall()

class SetupAltoInstall(install):
    def run(self):
        install.run(self)
        os.system("echo 'Setting up Alto parser'")
        runScriptInstall()

class SetupAltoEgg(egg_info):
    def run(self):
        egg_info.run(self)
        os.system("echo 'Setting up Alto parser'")
        runScriptInstall()

def runScriptInstall():
    if os.name == 'nt': # use ps1 script
        runWindows()
    else:
        runUnix()

def runWindows():
    os.system(f'powershell iex -Command "$( get-content {os.getcwd()}/setup.ps1 | Out-String )"')

    if os.environ.get('ALTO_JAR') is None:
        alto_path = os.path.expanduser("~/tuw_nlp_resources/alto-2.3.6-SNAPSHOT-all.jar")
        os.system(f'SETX ALTO_JAR \"{alto_path}\"')

def runUnix():
    os.system(f"bash {os.getcwd()}/setup.sh")

    if os.environ.get('ALTO_JAR') is None:
        with open(os.path.expanduser("~/.bash_profile"), "a") as outfile:
            alto_path = os.path.expanduser("~/tuw_nlp_resources/alto-2.3.6-SNAPSHOT-all.jar")
            outfile.write(f"export ALTO_JAR={alto_path}")
            os.system(f'bash -c \'source {outfile}\'')

setup(
    name='tuw-nlp',
    version='0.1',
    description='NLP tools at TUW Informatics',
    url='http://github.com/recski/tuw-nlp',
    author='Gabor Recski,Adam Kovacs',
    author_email='gabor.recski@tuwien.ac.at,adam.kovacs@tuwien.ac.at',
    license='MIT',
    install_requires=[
        'dict-recursive-update',
        'networkx',
        'penman',
        'stanza==1.1.1',
        'nltk'
    ],
    packages=find_packages(),
    scripts=['setup.sh'],
    include_package_data=True,
    cmdclass={'develop': SetupAltoDevelop, 'install': SetupAltoInstall, "egg_info": SetupAltoEgg},
    zip_safe=False)
