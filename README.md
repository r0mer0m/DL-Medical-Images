# Deep Learning with Medical Images: Best Practices for Small Data

Code supporting the paper Deep Learning with Medical Images: Best Practices for Small Data

## Status: Dev

Currently re-writting the code on a series of notebooks to make this research more accessible and hopefully useful for new Deep Learning practicioners.

- [X] Training Methods
  - [X] ChestXray 14
  - [X] Pneumnoia
  - [X] Emphysema
  - [X] Hernia
- [X] Transfer Learning Methods
  - [X] ChestXray 14
  - [X] Pneumnoia
  - [X] Emphysema
  - [X] Hernia
- [X] Transfer Learning Up-stream Data-sets
  - [X] Up-stream models
    - [X] MURA
    - [X] CheXpert
    - [X] CheX-ray13 ( Emphysema )
  - [X] Down-stream models
    - [X] ChestXray 14
    - [X] Pneumnoia
    - [X] Emphysema
    - [X] Hernia
- [ ] Experiments on synthetic labels:
  - [ ] Labels creation (cnn/ridge).
  - [ ] Fitting on synthetic data.

## Introduction

Major advances in Deep Learning have influenced the way in which people do Medicine. After observing the promising results of projects such as Google's [Detection of Diabetic retinopathy](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html) or Stanford's leaderbord on [MURA](https://stanfordmlgroup.github.io/competitions/mura/) a large amount of medical practitioners have been starting to use the same models. 

One of the fields where that happend is [Medical Physics](https://medicalphysics.duke.edu/medical_physics). Unfortunately, the conditions of that field are not the same as the condition where those major projects were build upon (for the general practitioner). We are talking about the amount of available data. Most of those projects had a large financial investment on data collection that yield to obtain 128,175 retinal images and 40,561 multi-view images respectively.

## Data

We used the following data-sets:

- [**Chest X-rays 14**](https://www.kaggle.com/nih-chest-xrays/data)

- [**RSNA Bone Age**](https://www.kaggle.com/kmader/rsna-bone-age)

- [**ImageNet**](http://www.image-net.org/)

- [**MURA**](https://stanfordmlgroup.github.io/competitions/mura/)

- [**CheXpert**](https://stanfordmlgroup.github.io/competitions/chexpert/)

All the data used in this project is public to encourage reproducibility.

## Authors

[Miguel Romero](https://github.com/r0mer0m), [Yannet Interian, Ph.D.](https://www.usfca.edu/faculty/yannet-interian), [Gilmer Valdes, Ph.D.](https://radonc.ucsf.edu/gilmer-valdes) and [Timothy D. Solberg, Ph.D.](https://radonc.ucsf.edu/tim-solberg).

## Acknowledgments

We thank [Jeremy Howard](https://www.usfca.edu/faculty/jeremy-howard) for advisoring.

## Contact Information

`email: mromerocalvo@usfca.edu`
