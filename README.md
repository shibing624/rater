# rater

**rater** is a comparative framework for multimodal recommender systems. It was developed to facilitate the designing, comparing, and sharing of recommendation models.

## Feature
####


## Data


1. ml-1m: http://files.grouplens.org/datasets/movielens/ml-1m.zip
2. delicious-2k: http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip
3. lastfm-dataset-360K: http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz
4. slashdot: http://snap.stanford.edu/data/soc-Slashdot0902.txt.gz
5. epinions: http://snap.stanford.edu/data/soc-Epinions1.txt.gz
6. ml-100k: http://files.grouplens.org/datasets/movielens/ml-100k.zip
7. Criteo(dac full): https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
8. Criteo(dac sample): http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz

## Install
```
pip3 install rater
```

or

```
git clone https://github.com/shibing624/rater.git
cd rater
python3 setup.py install
```

## Usage


Load the built-in [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset (will be downloaded if not cached):



**Output:**

|                          |    MAE |   RMSE |    AUC | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |  Train (s) | Test (s) |
| ------------------------ | -----: | -----: | -----: | ------: | ------: | --------: | --------: | ---------: | -------: |
| [MF](cornac/models/mf)   | 0.7430 | 0.8998 | 0.7445 |  0.0479 |  0.0556 |    0.0352 |    0.0654 |       0.13 |     1.57 |
| [PMF](cornac/models/pmf) | 0.7534 | 0.9138 | 0.7744 |  0.0617 |  0.0719 |    0.0479 |    0.0880 |       2.18 |     1.64 |
| [BPR](cornac/models/bpr) |    N/A |    N/A | 0.8695 |  0.0975 |  0.1129 |    0.0891 |    0.1449 |       3.74 |     1.49 |

For more details, please take a look at our [examples](examples).

## Models

The models supported are listed below. Why don't you join us to lengthen the list?


## Contribute


Your contributions at any level of the library are welcome. If you intend to contribute, please:
  - Fork the rater repository to your own account.
  - Make changes and create pull requests.

You can also post bug reports and feature requests in [GitHub issues](https://github.com/shibing624/recommender/issues).

## License

[Apache License 2.0](LICENSE)



## Reference


* [Multilayer Perceptron Based Recommendation]
* [Autoencoder Based Recommendation]
* [CNN Based Recommendation]
* [RNN Based Recommendation]
* [Restricted Boltzmann Machine Based Recommendation]
* [Neural Attention Based Recommendation]
* [Neural AutoRegressive Based Recommendation]
* [Deep Reinforcement Learning for Recommendation]
* [GAN Based Recommendation]
* [Deep Hybrid Models for Recommendation]
* [maciejkula/spotlight](https://github.com/maciejkula/spotlight)
* [shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR)
* [推荐系统实践]
* [Magic-Bubble/RecommendSystemPractice](https://github.com/Magic-Bubble/RecommendSystemPractice)