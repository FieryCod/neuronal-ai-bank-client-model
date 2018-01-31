# Neuronal Bank Client Model
Trained by Multi-layer Perceptron classifier.

## Prerequisites

1. Install `pipenv` globally using `pip`
  - ```bash
    $ pip install pipenv
    ```
    [More informations](https://github.com/pypa/pipenv#installation)
 
  - When you encounter problems like: `pew` is not found in your path 
  
    ```bash
    # First uninstall pipenv, virtualenv and pew:
    $ pip uninstall virtualenv pew pipenv
    # Then install it again in that order
    $ pip install virtualenv pew pipenv
    ```

2. Make sure that `pipenv` is in your $PATH. Add following line to shell configuration file: (bash: `.bashrc`, zsh: `.zshrc`)
    ```bash
    export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
    ```

3. Install dependencies:
    ```bash
    $ pipenv install
    ```

4. Run the project
    ```bash
    $ ./run-project
    ```

## Development ##

In order to lint the project use
```bash 
$ ./lint-project
```

For switching to virtualenv shell use
```bash
$ pipenv shell
```

## Issues ##

If you got a `ImportError` please install `python3-tk` module

- Ubuntu:
    ```bash
    $ sudo apt-get install python3-tk
    ```

## Resources ##

[ML - Bank Marketing Solution](https://www.kaggle.com/mayurjain/ml-bank-marketing-solution "Bank Marketing Solution")

[Preprocessing Bank Marketing dataset](https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5 "Preprocessing Bank Marketing dataset")

[Bank marketing data-set information](https://archive.ics.uci.edu/ml/datasets/bank+marketing "Bank marketing dataset information")
