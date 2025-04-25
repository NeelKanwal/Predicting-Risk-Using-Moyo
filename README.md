![image](https://github.com/user-attachments/assets/041fedb4-cfd1-4ca9-b40d-5c1f629436a5)# A-Self-supervised-Transformer-Approach-for-Fetal-Heart-Rate-Inpainting-and-Forecasting

This repository is the official Pytorch implementation for AI-powered inpainting and forecasting for fetal heart rate signals: 

## Description

Fetal heart rate (FHR) monitoring is crucial for evaluating fetal well-being during prenatal care. Recent advances in monitoring technologies, such as wearable FHR monitors, now allow for easy fetal movement analysis in resource-constrained environments without compromising maternal mobility. However, sensor displacement during free movement or changes in fetal and maternal position often results in signal dropouts, creating gaps in the obtained FHR data. These missing signals impede the extraction of meaningful insights and hinder automated downstream analysis. In this paper, we propose a novel approach using masked transformer-based autoencoders to learn signal reconstruction by focusing on both spatial and frequency components of the data. Our proposed method demonstrates robustness in handling diverse durations of missing data, contributing significantly to the field by performing accurate imputations to complete missing data and forecasting FHR signals. The developed models can be integrated into wearable FHR monitoring devices, creating a robust tool for handling missing samples and improving data integrity. This enhancement enables more reliable downstream analysis, potentially leading to earlier detection of fetal distress.

![image](https://github.com/user-attachments/assets/1570e5fe-4147-4006-91fa-7275228d2e41)


# Requirements
- Histolab
- Pytorch
- Pandas
- Numpy
- Scikit-learn
- MMCV
- Scipy

Use requirement.txt for exact versions. 

### Results
![image](https://github.com/user-attachments/assets/c474fdda-7e96-453d-940e-8156cd2a7191)


Model weights:
Model weights used in the paper are available at: https://drive.google.com/drive/folders/1zYHzgeG6IsYNvQW-QHVdnj--9oet4gWi?usp=sharing


## Citing
If you use or find this code repository useful, consider citing it as follows:
```@misc{}}

```





