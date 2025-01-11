# Sentimental Analysis Using Parallel DeepLearning Using PyTorch


### Sentiment Analysis on Amazon Customer Reviews Using Parallel Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Goal](#goal)
4. [Dataset Description](#dataset-description)
5. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Parallelization Techniques](#parallelization-techniques)
6. [Results and Performance Matrix](#results-and-performance-matrix)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction
In today’s digital era, online reviews play a critical role in consumer decision-making. This project focuses on **sentiment analysis** of Amazon customer reviews using **parallel deep learning techniques**, enabling businesses to derive actionable insights efficiently from vast datasets.

## Motivation
With the explosive growth of online reviews, traditional data analysis techniques often struggle with large-scale, unstructured data. This project addresses these challenges through parallel computing and machine learning, ensuring scalability and timely insights.

## Goal
Develop a scalable, efficient sentiment analysis system capable of processing millions of Amazon customer reviews, enabling businesses to:
- Understand consumer sentiment.
- Identify trends in feedback.
- Optimize decision-making strategies.

## Dataset Description
The dataset comprises **4 million Amazon customer reviews**:
- **Training data:** 3.6 million records.
- **Test data:** 0.4 million records.
- Reviews are labeled as:
  - `__label__1` (negative).
  - `__label__2` (positive).

Neutral reviews were excluded for binary sentiment analysis.

## Methodology

### Data Preprocessing
- **Steps Taken**:
  - Extraction of sentiment labels.
  - Conversion to binary labels (`1` for positive, `0` for negative).
  - Text cleaning (lowercasing, removal of special characters, etc.).
  - Handling null values and balancing the dataset.
- **Tools Used**:
  - Dask for distributed processing.
  - NLTK for tokenization.

### Parallelization Techniques
1. **Distributed Data Parallel (DDP)**:
   - Leveraged PyTorch's DDP for synchronized gradient updates across CPUs.
   - Improved performance with up to **28 CPUs**.
   - Achieved **95.17% accuracy**.

2. **DDP with Model Parallelism**:
   - Distributed model layers across devices for optimized memory usage.
   - Combined data and model parallelism for handling large datasets.

3. **Mixed Precision Training**:
   - Used PyTorch's AMP for reduced memory and computation requirements.

---

## Results and Performance Matrix

### Key Metrics
- **Training Time**: Measured in minutes.
- **Accuracy**: Percentage of correctly classified sentiments.
- **Loss**: Binary cross-entropy loss for classification.

### Performance Matrix

| **CPUs Used** | **Approach**               | **Accuracy** | **Loss** | **Training Time (mins)** |
|---------------|----------------------------|--------------|----------|--------------------------|
| 8             | DDP                        | 94.02%       | 0.0209   | 263                      |
| 12            | DDP                        | 93.79%       | 0.0141   | 197                      |
| 16            | DDP                        | 93.40%       | 0.0100   | 162                      |
| 20            | DDP                        | 92.95%       | 0.0090   | 150                      |
| 28            | DDP                        | **95.17%**   | **0.0040** | **135**                  |
| 8             | DDP + Model Parallel       | 93.72%       | 0.0225   | 248                      |
| 12            | DDP + Model Parallel       | 93.55%       | 0.0140   | 194                      |
| 16            | DDP + Model Parallel       | 93.96%       | 0.0100   | 180                      |
| 20            | DDP + Model Parallel       | 82.30%       | 0.0940   | 173                      |
| 28            | DDP + Model Parallel       | 94.54%       | 0.0040   | 183                      |
| 4             | Mixed Precision (GPU + CPU)| 91.45%       | 0.1350   | 151                      |
| 8             | Mixed Precision (GPU + CPU)| 91.84%       | 0.0660   | 137                      |
| 12            | Mixed Precision (GPU + CPU)| 91.48%       | 0.0440   | 135                      |

---

## Conclusion
This project demonstrates the power of parallel processing in sentiment analysis:
- **Distributed Data Parallel (DDP)** achieved the best accuracy and training time.
- Combining **DDP with Model Parallelism** balanced memory and computation but introduced synchronization overheads.
- **Mixed Precision Training** reduced computational costs, making it suitable for GPU-accelerated environments.

By leveraging parallelization techniques, we achieved a scalable, high-performance sentiment analysis pipeline that efficiently processed millions of reviews.
