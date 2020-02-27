# Erich Relation Classification

(Unofficial) **Reproduction** of the paper `R-BERT`: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>


## Dependencies

- perl (For evaluating official f1 score)
- python>=3.6
- torch==1.1.0
- pytorch-transformers==1.1.0
- tensorboardX
- tensorflow-gpu/tensorflow==1.15.2
- scikit-learn


## How to run

```bash
$ bash preprocess.sh
$ bash train.sh
```

- Prediction will be written on `eval_result.txt` in `dataset/semeval` directory.

## Official Evaluation

```bash
$ bash test.sh
# single GPU 2080Ti
# bert base en_uncased model: macro-averaged F1 = 89.16%
# bert big en_uncased model(fp16): macro-averaged F1 = 89.63%(not submitted in paper)
# bert big en_uncased(Whole Word Masking)(fp16): macro-averaged F1 = 89.8%(not submitted in paper)
```
- if you want to use fp16, you need [apex](https://github.com/NVIDIA/apex)(GCC4.9) and [torch](https://download.pytorch.org/whl/cu100/torch_stable.html)(cuda10)
- Evaluate based on the official evaluation perl script.
  - MACRO-averaged f1 score (except `Other` relation)
