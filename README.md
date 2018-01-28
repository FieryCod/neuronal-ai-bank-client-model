# Neuronal-ai-bank-client-model #

-------------------------------------------------------------------------------

## Prerequisites ##

1. Install `pipenv`
  - `pip install --user pipenv`

2. Make sure that `pipenv` is in your $PATH. Add this somewhere according to the shell
   you use (`.zprofile` or `.bashrc` or `.zsrc` or `.zshenv`)
  - `export PATH=$HOME/.local/bin:/usr/local/bin:$PATH`

3. Install dependencies:
  - `pipenv install`

4. Run the project
  - `sh run-project`


## Development ##

In order to lint the project use
  - `sh lint-project`

For switching to virtualenv shell use
  - `pipenv shell`

## Issues ##

If you got a `ImportError` please install `python3-tk` module

`sudo apt-get install python3-tk`

## Resources ##

[ML - Bank Marketing Solution](https://www.kaggle.com/mayurjain/ml-bank-marketing-solution "Bank Marketing Solution")

[Preprocessing Bank Marketing dataset](https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5 "Preprocessing Bank Marketing dataset")

[Bank marketing dataset information](https://archive.ics.uci.edu/ml/datasets/bank+marketing "Bank marketing dataset information")
