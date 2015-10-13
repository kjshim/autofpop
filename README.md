Auto solver for friends pop

# Prerequisites

* adb
* python 2.7 and related libraries (See requirements.txt)

# Setup (on Mac)

## Install adb

```
$ brew install android-platform-tools
```

## Install conda with python 2.7

```
$ wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
$ bash Miniconda-latest-MacOSX-x86_64.sh
```

## Install requirements with conda

```
$ conda install -y --file requirements.txt
```

# Usage

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

* andlib.py
* ScreenReader.py
* recognition.py
* friendspop.py
* test_friendspop.py
