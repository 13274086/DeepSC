# Deep Learning Enabled Semantic Communication Systems

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of  Deep learning enabled semantic communication systems.

## Requirements
+ See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.

## Bibtex
```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={2021},
  volume={Early Access}}
```
## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## Train
```shell
python main.py 
```
### Notes
+ Please carefully set the $\lambda$ of mutual information part since I have tested the model in different platform, 
i.e., Tensorflow and Pytorch, same $\lambda$ shows different performance.  

## Evaluation
```shell
python performance.py
```
### Notes
+ If you want to compute the sentence similarity, please download the bert model.
