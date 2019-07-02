#!/usr/bin/env python

import setuptools
import subprocess

try:
    cp = subprocess.run(['musescore', '--version'])
    has_musescore = cp.returncode == 0
except:
    has_musescore = False
assert has_musescore, 'MuseScore is not installed. install it and set path.'

setuptools.setup(
    name='nmftranscription',
    version='0.1',
    description='',
    author='',
    author_email='',
    url='https://github.com/arity-r/nmf-transcription',
    packages=['nmftranscription'],
    python_requires='~=3.6',
    # TODO: specify version
    install_requires=[
        'numpy',
        'h5py',
        'librosa',
        'mido',
    ],
    package_dir={'nmftranscription': 'nmftranscription'},
    scripts=[
        'scripts/nmftranscription-prepare.py',
        'scripts/nmftranscription-separate.py',
        'scripts/nmftranscription-restore.py',
    ],
)
