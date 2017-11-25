# Artificial Neuronal Network Playground (Python 3)

Inspired from several tutorials about Neuronal Networks I created this Python project. Here is a list with the tutorial sources:

Modul             | What it does                         | Tutorial
---               | ---                                  | ---
Backpropagation   | Learn a simple logic function        | [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/)
RNN               | Learn how to add two binary numbers  | [Anyone Can Learn To Code an LSTM-RNN in Python](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)

---
## Setup

In order to be able to run and test your `python` library/application make sure to complete the setup. The setup assumes that you have successfully installed `python` and `pip`. I strongly encourage to install a current version.

### 1. Create an virtual environment

First of all I recommend to create and activate an isolated Python environment with [`virtualenv`](https://virtualenv.pypa.io/en/stable/). You can also skip this step but then all python dependencies will be installed global.

There are many ways to install `virtualenv`. I like to use [`autovenv`](https://autovenv.readthedocs.io/en/latest/) for that purpose.

### 2. Fetch dependencies

When your environment is activated then get the project's dependencies by running:

```bash
pip install -r requirements.txt
```

<!-- ### 3. Install the project

Before running any script or tests it is necessary to install the current module `project` with pip in "editable" mode. Otherwise `python` or `pytest` can't refer to the project itself and would stop with an Exception like this: `ImportError: No module named 'project'`. Install the `project` from current directory:

```bash
pip install -e .
```

Now you can develop your code and execute it. -->

---
## Run the application

Run one of the modules like:
```bash
python src/rnn/rnn.py
```

<!-- ---
## Testing - `pytest`

This project uses `pytest` for testing your library/application. As you can see in the `requirements.txt` it does not come as standard package with `python` and has to be installed in before. After installing `pytest` you should be able to use `pytest` as command line tool.

#### Running tests

Run all tests:

```bash
pytest tests/
``` -->
