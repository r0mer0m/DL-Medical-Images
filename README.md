# Deep Learning and Medical Physics: When small is too small?

Code supporting the paper [Deep Learning and Medical Physics: When small is too small?](www.placeholder.com)(placeholder)

## Progress

Currently re-writting the code on a series of notebooks to make this research more accessible and hopefully useful for new Deep Learning practicioners.

- [X] Source data.
- [X] Training MURA weights.
- [ ] Experiments on real data: 
  - [ ] Multilabel task: [Chest X-ray 14](https://www.kaggle.com/nih-chest-xrays/data).
	  - [ ] First training approach.
	  - [ ] Second training approach.
	  - [ ] Traditional approach.
  - [ ] Binary task: [Pneumonia vs non-pneumonia](https://www.kaggle.com/nih-chest-xrays/data).
    - [ ] First training approach.
    - [ ] Second training approach.
    - [ ] Traditional approach.
- [ ] Experiments on synthetic labels:
  - [ ] Labels creation (cnn/ridge).
  - [ ] Fitting on synthetic data.

## Introduction

Major advances in Deep Learning have influenced the way in which people do Medicine. After observing the promising results of projects such as Google's [Detection of Diabetic retinopathy](https://ai.googleblog.com/2016/11/deep-learning-for-detection-of-diabetic.html) or Stanford's leaderbord on [MURA](https://stanfordmlgroup.github.io/competitions/mura/) a large amount of medical practitioners have been starting to use the same models. 

One of the fields where that happend is [Medical Physics](https://medicalphysics.duke.edu/medical_physics). Unfortunately, the conditions of that field are not the same as the condition where those major projects were build upon (for the general practitioner). We are talking about the amount of available data. Most of those projects had a large financial investment on data collection that yield to obtain 128,175 retinal images and 40,561 multi-view images respectively.

## Data

We used the following data sets:

**Chest X-rays 14**: This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.). [Link](https://www.kaggle.com/nih-chest-xrays/data).

**RSNA Bone Age**: This RSNA Bone Age dataset is composed of 12 thousand hand X-ray images with age labels in months. The images belong to children and had been annoymized for research purposes. The dataset was originally published on CloudApp as an RSNA challenge. [Link](https://www.kaggle.com/kmader/rsna-bone-age).

**ImageNet**: ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The entire dataset contains 14,197,122 images and has been used in this project for transfer learning. [Link](http://www.image-net.org/).

**MURA**: MURA (musculoskeletal radiographs) is a large dataset of bone X-rays. Algorithms are tasked with determining whether an X-ray study is normal or abnormal. The dataset contains 40,561 multi-view X-ray images from different parts of the body and was used in this study for transfer learning. [Link](https://stanfordmlgroup.github.io/competitions/mura/).

All the data used in this project is public to encourage reproducibility.

## Authors

[Miguel Romero](https://github.com/r0mer0m), [Yannet Interian, Ph.D.](https://www.usfca.edu/faculty/yannet-interian), [Gilmer Valdes, Ph.D.](https://radonc.ucsf.edu/gilmer-valdes) and [Timothy D. Solberg, Ph.D.](https://radonc.ucsf.edu/tim-solberg).

## Acknowledgments

We thank [Jeremy Howard](https://www.usfca.edu/faculty/jeremy-howard) for advisoring. 
