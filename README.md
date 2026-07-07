# Prognostix: Explainable Deep Learning AI for Industrial Bearing Fault Diagnosis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

*Predictive maintenance of industrial bearings using attention-based deep learning with explainability*

Live Demo - https://prognostix.streamlit.app/

</div>

---

## Overview

Prognostix is an advanced deep learning framework designed for real-time prediction and early detection of bearing faults in industrial machinery. The system combines the strengths of Convolutional Neural Networks for feature extraction, Bidirectional LSTM for temporal sequence modeling, and MultiheadAttention mechanisms for interpretable focus on critical signal regions.

By integrating three explainability methods (SHAP, Attention Visualization, and Temporal Saliency), Prognostix moves beyond black-box predictions to provide actionable insights into why the model identifies specific fault conditions. This is critical for industrial maintenance teams that need to trust and validate AI-driven decisions.

---

## Key Features

- **Multi-Scale Feature Extraction**: CNN with variable kernel sizes (3, 5) captures both high-frequency noise and low-frequency bearing fault signatures
- **Temporal Sequence Modeling**: Bidirectional LSTM with 2 layers processes forward and backward temporal dependencies
- **Interpretable Attention**: MultiheadAttention (4 heads) reveals temporal regions critical to predictions
- **Four-Class Classification**: Normal, Ball Fault, Inner Race Fault, Outer Race Fault
- **Explainability Suite**: 
  - SHAP DeepExplainer for global feature importance
  - Attention weight visualization for temporal focus
  - Temporal saliency for gradient-based sensitivity analysis
- **Robust Preprocessing**: Bandpass filtering (500-10000 Hz), StandardScaler normalization, SMOTE resampling for class imbalance
- **Production-Ready Deployment**: Streamlit web application with multi-page interface, sample data demo, and user upload support
- **Performance**: 98.08% accuracy, 0.9808 weighted F1 score, 0.9992 average ROC-AUC

---

## Model Architecture

### Deep Learning Model: AttentionBearingNet
 
A hybrid CNN + LSTM + Attention architecture designed for temporal signal classification:
 
```
Input Signal (2048 samples @ 48 kHz)
    ↓
CNN Feature Extraction (multi-scale kernels: 3, 5, 7)
    ↓
┌─────────────────────────────────┬──────────────────┐
│                                 │                  │
LSTM Branch                    Attention Branch      
(Bidirectional, 2 layers)      (4 heads, temporal focus)
(hidden=64)                    (self-attention)
│                                 │
└─────────────────────────────────┴──────────────────┘
                    ↓
            Fusion Layer (concat)
                    ↓
        Classification Head (4 classes)
        Uncertainty Head (epistemic)
```

**Total Parameters**: 1,874,052  
**Model Size**: ~7.2 MB  
**Framework**: PyTorch 2.0

---

## Dataset

### CWRU Bearing Dataset

**Source**: Case Western Reserve University | [Kaggle Link](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

**Signal Characteristics**:
- **Variable**: X098_DE_time (Drive-End acceleration)
- **Original Length**: 483,903 samples
- **Sampling Rate**: 48 kHz
- **Duration per Sample**: ~10 seconds

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 98.08% |
| **Weighted F1 Score** | 0.9808 |
| **Average ROC-AUC** | 0.9992 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 1.0000 | 1.0000 | 1.0000 | 71 |
| Ball Fault | 0.9626 | 0.9671 | 0.9649 | 213 |
| Inner Race Fault | 1.0000 | 1.0000 | 1.0000 | 285 |
| Outer Race Fault | 0.9670 | 0.9624 | 0.9647 | 213 |

### ROC-AUC Scores (One-vs-Rest)

- Normal: 1.0000
- Ball Fault: 0.9983
- Inner Race Fault: 1.0000
- Outer Race Fault: 0.9983

---

## Explainability & Interpretability
 
Prognostix implements a multi-method XAI approach, providing both research-grade analysis and production-ready interpretability:
 
### Phase 1: Research & Training (Colab Notebook)
 
**SHAP (SHapley Additive exPlanations)**
- Post-hoc model explanation using Shapley values
- Analyzes feature contributions to predictions
- Identifies critical time windows and frequency regions
- Perfect for: Understanding what the model learned during training
- Computational cost: Higher (but justified for research)
### Phase 2: Production & Deployment (Streamlit App)
 
Three complementary XAI methods, optimized for real-time inference:
 
#### 1. **Attention Visualization**
- Directly visualizes learned temporal focus
- Shows which time regions the model attends to
- Interpretable: 4-head attention weights
- Speed: Instant (no extra computation)
#### 2. **Temporal Saliency Maps**
- Gradient-based input sensitivity analysis
- Computes partial derivatives: ∂(logits)/∂(input)
- Reveals sharp peaks (impulse events) and broad regions (sustained patterns)
- Speed: Real-time (gradient computation)
#### 3. **Feature Importance (Perturbation-Based)**
- Sensitivity analysis through input masking
- Measures prediction change when signal segments are zeroed
- Cumulative importance across time steps
- Speed: Fast (no gradient computation needed)
**Why Three Methods?**
- Different perspectives reveal complementary insights
- Attention = learned focus
- Saliency = gradient-based sensitivity  
- Perturbation = empirical importance
- Triangulation increases confidence in explanations
---

## Installation & Setup
 
### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
### Step 1: Clone Repository
```bash
git clone https://github.com/Khushisoni702/Prognostix.git
cd PROGNOSTIX
```
 
### Step 2: Create Virtual Environment
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS/Linux
source venv/bin/activate
```
 
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
 
### Step 4: Run Streamlit App
```bash
streamlit run app.py
```
 
The app will open at `http://localhost:8501`
 
---

## Project Structure

```
Prognostix/
├── app.py                           # Streamlit application
├── best_model.pth                   # Trained model weights
├── prognostix_scaler.pkl            # StandardScaler for preprocessing
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
├── Prognostix_Complete_Training.ipynb  # Training notebook (reference)
├── samples/
│   ├── demo_normal.mat     # Sample normal bearing signal
│   ├── demo_ball.mat       # Sample ball fault signal
│   ├── demo_inner.mat      # Sample inner race fault signal
│   └── demo_outer.mat      # Sample outer race fault signal

```

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Contact & Support

**Author**: Khushi Soni  
**GitHub**: [@Khushisoni702](https://github.com/Khushisoni702)  

For questions, issues, or collaborations, please open an issue on GitHub or contact via LinkedIn.

---

## Roadmap

Future enhancements:

- [ ] Multi-bearing system monitoring
- [ ] Real-time anomaly detection stream
- [ ] Custom model fine-tuning interface
- [ ] Integration with industrial IoT platforms
- [ ] Advanced signal preprocessing options
- [ ] Comparative model benchmarking
- [ ] Mobile app deployment
- [ ] API endpoint for production integration

---

## Disclaimer
 
Prognostix is a research and development tool. While it demonstrates state-of-the-art performance on the CWRU dataset, real-world bearing diagnostics should incorporate:
- Domain expert validation
- Multiple sensor inputs
- Operational context
- Maintenance history
- Manufacturer specifications

Always combine AI predictions with professional engineering judgment.

---
 
**Last Updated**: 2024
**Status**: Production Ready

