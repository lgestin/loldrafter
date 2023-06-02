# LOLDRAFT

loldraft is a side project training transformer models to draft team compositions for the game League of Legends.

The drafting process is when the players chose which champion they are going to play during the game. It usually happens in a finite set of phases where each of the teams ban and picks champions. 

This project aims to simulate this process and train the model to predict which champion the current team should predict given the current draft state (picks/bans).

## Data Scraping

The data used for the project is all the pro games (regular season and playoffs) that happened between 2019 and today in the major leagues. 

The data used to train the models was scraped from the internet. It can be downloaded again using the command:
```
python scraper/scrape.py --dump_path path/to/dump/data
```
## Model training

The model can be trained using the following command:

```
python drafter/train.py --data_path path/to/data/data.json --exp_path path/to/exp
```
Most of the parameters can be changed when calling the script. 
The script will create a tensorboard at exp_path from which you can follow the experiment. At the moment model saving/loading isn't supported

## Runing using docker-compose

You can run this code if you have docker-compose installed. Be aware that you'll need to change the permissions of the files created to access them.

for scraping:
```
PATH_TO_DATA=/path/to/data docker-compose up scraping
```

for training:
```
PATH_TO_DATA=path/to/data PATH_TO_EXP=/path/to/exp docker-compose up training
```