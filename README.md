# Vietnamese Named Entity Recognition

![](https://img.shields.io/badge/F1-88.6%25-red.svg)

[[English](README.md)] [[Vietnamese](README.vi.md)]

This repository contains experiments in **Vietnamese Named Entity Recognition** problem. It is a part of [underthesea](https://github.com/magizbox/underthesea) project.

## Tables of contents

* [Installation](#installation)
  * [Requirements](#requirements)
  * [Download and Setup Environement](#download-and-setup-environment)
* [Usage](#usage)
  * [Train a new dataset](#train-a-new-dataset)
  * [Using a pretrained model](#using-a-pretrained-model)
  * [Sharing a model](#sharing-a-model)
* [Citation](#citation)

## Installation

### Requirements

* `Operating Systems: Linux (Ubuntu, CentOS), Mac`
* `Python 3.5+`
* `languageflow==1.1.7`

### Download and Setup Environment

Clone project using git

```
$ git clone git@github.com:magizbox/underthesea.ner.git
```

Create environment and install requirements

```
$ cd ner
$ conda create -n uts.ner python=3.5
$ pip install -r requirements.txt
```

## Usage

### Train a new dataset

```
$ cd ner
$ source activate ner
$ python train.py
```

### Using a pretrained model

To be updated

### Sharing a model

To be updated

## Citation

To be updated

Last update: 05/2018
