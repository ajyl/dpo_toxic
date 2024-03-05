# Mechanistically Understanding DPO: Toxicity

This repository provides the models, data, and experiments used in A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity.

## Models, Data

You can download the models and datasets used in our paper [here](https://drive.google.com/drive/folders/1baArqcjIc2Q4OllLVUz1hp3p3XxmdteK?usp=drive_link).

Save the checkpoints under `./checkpoints` and unzip the data files under `./data`.

## Experiments

All of our experiments can be found under `./toxicity`.
To run interventions, see `./toxicity/eval_interventions/run_evaluations.py`.

To re-create any of our figures, see `./toxicity/eval_interventions/figures`.

## Training DPO

To train your own dpo model:
```
cd toxicity/train_dpo
python train.py exp_name="[name of your experiment]"
```


## How to Cite

If you find our work relevant, please cite as following:

```
@article{lee2024mechanistic,
  title={A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity},
  author={Lee, Andrew and Bai, Xiaoyan and Pres, Itamar and Wattenberg, Martin and Kummerfeld, Jonathan K and Mihalcea, Rada},
  journal={arXiv preprint arXiv:2401.01967},
  year={2024}
}
```
