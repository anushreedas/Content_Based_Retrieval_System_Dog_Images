# Content-Based Image Retrieval System

A comparative study of four feature extraction approaches for building an image search engine — given a query image of a dog, the system retrieves the most visually similar images from a dataset of 133 dog breeds. Each approach captures a different notion of visual similarity: color, texture, edge orientation, and deep semantic features.

**Authors:** Anushree Das · Nikhil Yadav

**Dataset:** [Udacity Dog Breed Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) — 6,681 training / 835 validation / 836 test images across 133 breeds

---

## What it does

Rather than matching exact pixel values, CBIR systems find images that are *similar* to a query image based on extracted visual features. This project implements and compares four different feature extractors, each encoding a different aspect of what makes two images look alike:

| Approach | What it captures | Similarity metric |
|---|---|---|
| Color Histogram | Color distribution of the subject | Euclidean distance |
| Gabor Filter | Texture and edge orientation patterns | Euclidean distance |
| HOG (Histogram of Oriented Gradients) | Shape and edge structure | Euclidean distance |
| ResNet50 (fine-tuned) | Deep semantic features | Class-based retrieval |

All four approaches first **segment the dog from the background** using a pretrained segmentation model before extracting features, so retrieval is based on the dog's visual properties rather than confounding background colors or textures.

---

## Approach overview

### Shared preprocessing — background segmentation

Each image passes through a pretrained segmentation model that produces a foreground mask isolating the dog from its background. This mask is applied before any feature extraction, so a black dog photographed against a green lawn and a black dog photographed in a studio both produce similar feature vectors based on the dog itself rather than the setting.

### 1. Color Histogram

The segmented dog image is passed to a color histogram descriptor that computes the distribution of pixel intensities across color channels. The histogram is normalized to achieve scale invariance (a small and large image of the same dog produce comparable histograms). Features for all training images are precomputed and stored in a pickle file.

At query time: segment the query image → compute its normalized color histogram → calculate Euclidean distance against all stored histograms → return the k nearest images.

**What it retrieves:** images of dogs with similar coat colors and color patterns.

### 2. Gabor Filter

A Gabor kernel (size 21×21, Gaussian standard deviation 8.0, orientation 45°, wavelength 10) is applied to the segmented and resized (128×128) dog image. Gabor filters are bandpass filters that capture energy at specific spatial frequencies and orientations — effectively encoding texture and the local directionality of edges.

At query time: segment → apply Gabor filter → compute Euclidean distance against precomputed Gabor features.

**What it retrieves:** images of dogs in similar poses and postures, since body orientation dominates the texture feature space.

### 3. Histogram of Oriented Gradients (HOG)

HOG features are extracted using OpenCV's `HOGDescriptor` on the segmented dog image. HOG divides the image into cells, computes a histogram of gradient directions within each cell, and concatenates these histograms into a feature vector. This captures edge structure and shape information more precisely than the Gabor filter.

At query time: segment → compute HOG descriptor → Euclidean distance against precomputed HOG features.

**What it retrieves:** images of dogs in similar positions and with similar body shapes. Slightly more precise than Gabor for pose-based retrieval.

### 4. ResNet50 (fine-tuned, transfer learning)

A ResNet50 pretrained on ImageNet is fine-tuned on the dog breed dataset by retraining its fully connected layers for 4 epochs using Adam optimizer and cross-entropy loss. The model achieves **84% test accuracy** across 133 breeds, with a validation loss of 0.503 after the final epoch.

At query time: the query image is classified into one of the 133 breed classes, and sample images from that class are returned. This is class-based retrieval rather than feature-distance retrieval — the full power of the network's learned representations is used to determine which breed the query belongs to.

**What it retrieves:** images of the same dog breed, with the highest visual coherence of all four approaches.

---

## Results

| Approach | Retrieval quality | Characteristic behavior |
|---|---|---|
| Color Histogram | Moderate | Retrieves dogs with matching coat color regardless of breed or pose |
| Gabor Filter | Moderate | Groups dogs by pose and body orientation; less sensitive to color |
| HOG | Good | Better pose matching than Gabor; more precise edge-based similarity |
| **ResNet50** | **Best** | Retrieves same-breed dogs; highest semantic coherence |

The neural network outperforms all classical feature extractors because it learns a richer representation that combines color, texture, shape, and higher-level semantic information — rather than capturing only one of these dimensions in isolation. The classical approaches remain useful baselines that are interpretable, require no training, and run without a GPU.

### Result for image retrieval based on Color Histogram
![Result for Color Histogram](https://github.com/anushreedas/Content_Based_Retrieval_System_Dog_Images/blob/main/dogImages/result/Color_result.png)

### Result for image retrieval based on texture obtained after applying Gabor Filter
![Result for Gobor Filter](https://github.com/anushreedas/Content_Based_Retrieval_System_Dog_Images/blob/main/dogImages/result/Gabor_result.png)

### Result for image retrieval based on HOG
![Result for HOG](https://github.com/anushreedas/Content_Based_Retrieval_System_Dog_Images/blob/main/dogImages/result/HOG_result.png)

### Result for image retrieval using Resnet50 NN
![Result for NN](https://github.com/anushreedas/Content_Based_Retrieval_System_Dog_Images/blob/main/dogImages/result/NN_result.png)

---

## Dataset

```
dogImages/
├── train/      # 6,681 images — used to build the feature index and fine-tune ResNet50
├── valid/      # 835 images
└── test/       # 836 images
```

133 subdirectories, one per breed. Each subdirectory is named `NNN.Breed_Name` where `NNN` is a zero-padded breed index.

---

## Requirements

```
Python 3.7+
torch
torchvision
opencv-python
numpy
scikit-image
tensorflow       # for segmentation model
matplotlib
pickle
```

---

## Academic context

Final project for **CSCI-631 Foundations of Computer Vision**, Rochester Institute of Technology, Spring 2021.

Demonstrates: classical feature extraction (color histograms, Gabor filters, HOG), transfer learning with ResNet50, foreground segmentation, Euclidean distance-based nearest-neighbour retrieval, and feature indexing with pickle for efficient search.

---

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://anushree-das.medium.com)
