'resnet101': '~/.cache/torch/checkpoints/resnet101s-03a0f310.pth',

# semantic-segmentation-codebase
Here is a pytorch codebase for semantic segmentation experiments.


## Installation
- Download the repository.
```
git clone https://github.com/YudeWang/semantic-segmentation-codebase.git
```
- Install python dependencies.
```
pip install -r requirements.txt
```
- Create softlink to your dataset. Make sure that the dataset can be accessed by `$your_dataset_path/VOCdevkit/VOC2012...`
```
ln -s $your_dataset_path data
```

## Experiments
Our experiments are placed in `.experiment/` and here are some implementations:
- DeepLabv3+ on VOC2012 dataset. [link](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/deeplabv3%2Bvoc)
- DeepLabv1 retrain on [SEAM](https://github.com/YudeWang/SEAM) pseudo labels. [link](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc)
- Coming soon...


Download resnet101 pretrained file on pretrained/ directory
Donwload link is in lib/net/backbone/resnet.py