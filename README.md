[![Build Status](https://travis-ci.org/kjshim/autofpop.svg?branch=master)](https://travis-ci.org/kjshim/autofpop)

Auto solver for friends pop

# Prerequisites

* adb
* python 2.7 and related libraries (See requirements.txt)
* keras (edge version for [this commit](https://github.com/fchollet/keras/commit/31cf6b16f48d1da338c7af26d64f5104534fe0ab))

# Setup (on Mac)

## Install adb

```
$ brew install android-platform-tools
```

## Install conda with python 2.7

```
$ wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
$ bash Miniconda-latest-MacOSX-x86_64.sh
$ rm Miniconda-latest-MacOSX-x86_64.sh
```

## Install requirements with conda

```
$ conda install -y --file requirements.txt
```

# Usage

* Make model

```
$ python -m autofpop.make_model # Models are stored at model/
```

* Run Friends pop
* Start the target stage
* Execute the program like blow

```
$ python test_friendspop.py
```

* Repeat
  * Wait for the result
  * Do just like the result
  * Close the result window

# Test

```
$ nosetests
```

# Files to check

* autofpop/andlib.py
* autofpop/ScreenReader.py
* autofpop/recognition.py
* autofpop/friendspop.py
* tests/test_friendspop.py
