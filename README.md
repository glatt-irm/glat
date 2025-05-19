# 🏥 Prostate Cancer Grading using GLAT & IRM
This repository contains the implementation of a **Graph Laplacian Attention-Based Transformer (GLAT)** integrated with an **Iterative Refinement Module (IRM)** for **prostate cancer grading** from whole-slide images (WSIs). 

### 🚀 **Key Features**
- **IRM** (Iterative Refinement Module) for **adaptive patch selection**, ensuring **only the most informative tissue regions** are selected.
- **GLAT** (Graph Laplacian Attention Transformer) **enforces spatial coherence**, preserving histological relationships.
- **Convex Aggregation** generates a **global WSI-level representation**, optimizing feature importance.
- **State-of-the-art performance** on **five public** and **one private dataset**.
- **Computationally efficient** while maintaining high accuracy.

---

<img src="./modelglat.jpg" width="400px" align="right" />

---

## 📂 **Project Structure**

---

## 📊 **Datasets**
We evaluated the model on **five public** and **one private dataset**:

| Dataset       | WSIs Count | Gleason Labels | Notes |
|--------------|------------|---------------|---------------------|
| **TCGA-PRAD** | 895 WSIs | Gleason Grading | Public dataset from TCGA |
| **SICAPv2** | 182 WSIs | Gleason Scores | High-quality annotations |
| **GLEASON19** | 331 TMAs | Pixel-level Annotations | Tissue Microarrays (TMAs) |
| **PANDA** | 12,625 WSIs | Primary & Secondary Gleason Grades | Largest dataset used |
| **DiagSet** | 430 WSIs | Prostate Cancer Grading | High-quality dataset |
| **Private Dataset** | 79 WSIs | Clinical-grade Annotations | Internal dataset |

---

## 🛠 **Preprocessing**
Patches are extracted using **([CLAM] (https://github.com/mahmoodlab/CLAM))** preprocessing pipeline:
- **Stain normalization**: Reduces staining variability across WSIs.
- **Tissue segmentation**: Removes irrelevant background regions.
- **Patch extraction**: Extracts **224×224** patches from WSIs.
- **Filtering**: Excludes patches with minimal tissue content.

---

## 🏗 **Installation**
To set up the environment, run:
```bash
cd ProstateCancerGrading
pip install -r requirements.txt

## 🛠 **Train the Model**
Run the following command to start training and evaluation:
```bash
python main.py


