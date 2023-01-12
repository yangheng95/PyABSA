# Data Preparation Tool

## Steps to annotate custom datasets for PyABSA
- 1. Pre-word segmentation
- 2. Annotate the segmented data
- 3. convert the apc dataset to atepc dataset
- 4. merge your custom dataset into integrated_datasets and PR

You can find the tool scripts at https://github.com/yangheng95/ABSADatasets/tree/v2.0/DPT

## Pre-word segmentation
Before annotating non-blank segmented text (e.g., English), you need to segment the data. Run `pre_word_segment_for_non_english_data.py` and
select the output file `*.seg` to annotate.

## A Stand-alone browser based tool to help process data for the training set

![image1](https://user-images.githubusercontent.com/4684417/139701633-d77a009b-1a12-4ef2-9663-37d2d36e1af1.JPG)

Once data saved, 3 files will be created:

1. a CSV file training set for classic sentiment analysis
2. a TXT file training set for PyABSA
3. a JSON file for saving unfinished work

#### The txt file generated for PyABSA will be structured as below:

![image](https://user-images.githubusercontent.com/4684417/139286711-152ea26e-5dbe-462a-bd73-287faf746572.png)
