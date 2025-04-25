# Fingerprint Matching with Siamese Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This project implements a Siamese network for fingerprint identification using the [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing), achieving **98.52% accuracy** on altered fingerprint matching.

## Key Features

- 🖼️ Processes 90×97 grayscale fingerprint images
- 🔍 Encodes fingerprints into 128-dimension normalized vectors
- 🤖 Uses triplet loss with cosine similarity
- 🚀 Achieves 98.5% accuracy on synthetic alterations
- 🛠️ Built with Trax and JAX for GPU acceleration

## Results
| Difficulty Level | Accuracy | Samples Tested |
|------------------|----------|----------------|
| All              | 98.52%   | 8,188          |
| Easy             | 99.30%   | 2,982          |
| Medium           | 98.42%   | 2,840          |
| Hard             | 97.68%   | 2,366          |

**Similarity Distribution**  
![Similarity Histograms](https://i.imgur.com/your-image-link.png)

## Model Architecture
```python
def Siamese(d_model=128):
    return tl.Serial(
        tl.Conv(16, (3,3), padding='VALID'),
        tl.MaxPool((3,3), padding='VALID'),
        tl.Conv(32, (3,3), padding='VALID'),
        tl.MaxPool((3,3), padding='VALID'),
        tl.Flatten(),
        tl.Dense(256),
        tl.Relu(),
        tl.Dense(384),
        tl.Relu(),
        tl.Dense(d_model),
        tl.Fn('Normalize', lambda x: x/tl.L2Norm(x))
    )
```

## Requirements
```bash
# For CUDA 12 (verify your CUDA version first)
pip install jax[cuda12_pip] trax==1.4.0 numpy pandas matplotlib opencv-python
```

## Usage
1. **Data Preparation**  
   Load SOCOFing dataset with altered fingerprints:
   ```python
   real = cv2.imread("1__M_Left_index_finger.BMP", cv2.IMREAD_GRAYSCALE)[2:-4, 2:-4]
   ```

2. **Training**  
   Train with triplet loss (12k steps recommended):
   ```python
   training_loop = train_model(model, TripletLoss, lr_schedule, 
                             train_generator, val_generator, 12000)
   ```

3. **Evaluation**  
   Match altered fingerprints to originals:
   ```python
   similarity = np.dot(query_encoded, stem_encoded.T)
   matched = np.argmax(similarity, axis=-1)
   ```

## Key Implementation Details
- **Triplet Loss**: Margin=0.25 with hard negative mining
- **Data Augmentation**: Random affine transformations
- **Batch Size**: 768 for optimal GPU utilization
- **Learning Rate**: Warmup + rsqrt decay schedule

## Dataset Structure
```
SOCOFing/
├── Real/               # 6,000 genuine fingerprints
├── Altered/
│   ├── Altered-Easy/   # 17,931 samples
│   ├── Altered-Medium/ # 17,067 samples
│   └── Altered-Hard/   # 14,272 samples
```

## Credits
- Dataset: [SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing) 
- Framework: [Trax](https://trax-ml.readthedocs.io/)
- Initial Inspiration: Coursera NLP Specialization

---

**Note**: This README corresponds directly to the Jupyter notebook implementation. For complete code and visualizations, see the notebook [here](your-kaggle-notebook-link).
