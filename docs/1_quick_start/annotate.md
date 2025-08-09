# Annotating Custom Datasets

This guide explains how to prepare and annotate your own datasets for use with PyABSA. The primary tool for this is the
Data Preparation Tool (DPT), which can be found in
the [ABSADatasets repository](https://github.com/yangheng95/ABSADatasets/tree/v2.0/DPT).

## Annotation Workflow

The process of annotating a custom dataset involves the following steps:

1. **Pre-segmentation**: Before annotating, your text needs to be segmented into words. For languages that are not
   space-delimited, you can use the `pre_word_segment_for_non_english_data.py` script provided in the DPT.
2. **Annotation**: Use the browser-based annotation tool to label your segmented text. This tool allows you to mark
   aspect terms and assign sentiment polarities.
3. **Conversion**: Once annotated, you can convert your dataset into formats suitable for different tasks, such as
   Aspect Polarity Classification (APC) or Aspect Term Extraction and Polarity Classification (ATEPC).
4. **Integration**: After preparation, you can merge your custom dataset with the integrated datasets in PyABSA to use
   it for training and evaluation.

## Using the Annotation Tool

The annotation tool is a standalone, browser-based application that simplifies the process of labeling your data.

![Annotation Tool Interface](https://user-images.githubusercontent.com/4684417/139701633-d77a009b-1a12-4ef2-9663-37d2d36e1af1.JPG)

When you save your work in the tool, it generates three files:

- A **CSV file** for classic sentiment analysis tasks.
- A **TXT file** formatted for use with PyABSA.
- A **JSON file** to save your progress and resume annotating later.

### PyABSA Data Format

The TXT file generated for PyABSA follows a specific format, where each line represents a sentence and its annotations.
The aspect terms are marked, and their sentiment polarity is assigned, as shown in the example below.

![PyABSA Data Format](https://user-images.githubusercontent.com/4684417/139286711-152ea26e-5dbe-462a-bd73-287faf746572.png)

By following this workflow, you can create high-quality, annotated datasets for your aspect-based sentiment analysis
tasks.
