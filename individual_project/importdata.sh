#!/bin/bash
# Export your Kaggle username and API key
# export KAGGLE_USERNAME=<YOUR USERNAME>
# export KAGGLE_KEY=<YOUR KAGGLE KEY>
# Data can also be downloaded here: https://kaggle.com/datasets/5150d9eaa3fdaa1c293c2a755ecc75662c56307e70037857a13230fa219febaf

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o ./twitter-community-notes.zip\
  https://www.kaggle.com/api/v1/datasets/download/albinlindqvist/twitter-community-notes

unzip twitter-community-notes.zip -d .
