[![Build Status](https://travis-ci.org/kjshim/autofpop.svg?branch=master)](https://travis-ci.org/kjshim/autofpop)

Auto solver for friends pop

# Prerequisites

* adb
* python 2.7 and related libraries (See requirements.txt)
* keras (edge version for [this commit](https://github.com/fchollet/keras/commit/31cf6b16f48d1da338c7af26d64f5104534fe0ab))
* tensorflow

# Setup (on Mac)

## Install adb

```sh
$ brew install android-platform-tools
```

## Install conda with python 2.7

```sh
$ wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
$ bash Miniconda-latest-MacOSX-x86_64.sh
$ rm Miniconda-latest-MacOSX-x86_64.sh
```

## Install requirements with conda

```sh
$ conda install -y --file requirements.txt
```

## Install TensorFlow

TODO: description

## Install Keras

edge version for [this commit](https://github.com/fchollet/keras/commit/31cf6b16f48d1da338c7af26d64f5104534fe0ab)

TODO: description

TODO: description for setting ~/.keras/keras.json

# Usage

## Make model

Skip this step if you want to use pre-generated model.

```sh
$ python -m autofpop.make_model # Models are stored at model/
```

## Check the accuracy of current model

```sh
$ python -m autofpop.show_accuracy
```

## Run

* Run Friends pop
* Start the target stage
* Execute the program like blow

```sh
$ python -m autofpop.run
```

* Repeat
  * Wait for the result
  * Do just like the result
  * Close the result window

# Test

```sh
$ nosetests
```

# Files to check

* autofpop/andlib.py
* autofpop/ScreenReader.py
* autofpop/recognition.py
* autofpop/friendspop.py
* tests/test_friendspop.py
