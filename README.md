# AI for Painting Analysis

> **Last update:** May 2026

This repository contains a multi-version machine learning project for analyzing paintings with image-based AI models. The goal is to build and compare systems that can classify artistic attributes from painting images, including visual tags such as subject, style, and color-related characteristics.

## Project overview
- Uses Python and TensorFlow/Keras to train multilabel image classification models.
- Works with painting image files and metadata stored in a local archive structure.
- Includes several experiment folders, such as AIP_v2, AIP_v3, and AIP_v4, each with its own preprocessing and training setup.
- Exports trained models in both Keras and TensorFlow Lite formats.

## Repository structure
- AIP_v2/: earlier experiments for image formatting and dataset preparation.
- AIP_v3/: label generation and multilabel training workflows.
- AIP_v4/: the current training pipeline with model export and evaluation scripts.

## Typical workflow
1. Prepare the painting dataset and place the images in the expected archive folder.
2. Build or update the multilabel CSV metadata file.
3. Run the training script from the relevant version directory.
4. Review metrics and export the trained model.

## Notes
This project is still experimental and intended mainly for learning, prototyping, and model development. The quality of results depends heavily on dataset quality, label accuracy, and preprocessing choices.

## Dataset
- WikiArt archive dataset: https://www.kaggle.com/datasets/steubk/wikiart?resource=download (dataset - no commercial use)
