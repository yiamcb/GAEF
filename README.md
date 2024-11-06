# GAEF: 
GAEF: Gated Recurrent Unit-Attentive EEG Fusion Model for Alzheimer's Disease Detection

The GAEF model is designed to enhance the accuracy of Alzheimer's Disease (AD) detection by effectively capturing both spectral and temporal dependencies in EEG data. The model incorporates advanced features and an attention mechanism to improve classification boundaries between AD, frontotemporal dementia (FTD), and cognitive normal (CN) cases.

# Code Components

**FeaturesExtraction.py:** This script is responsible for extracting a comprehensive set of features from EEG data. It includes methods to compute advanced features such as Higuchi fractal dimension, Lyapunov exponents, spectral entropy, and power spectral density across various EEG bands. These features help in capturing the complexity and irregularities in brain dynamics associated with neurological disorders.

**ModelArchitecture.py:** This file contains the complete architecture of the GAEF model. The architecture integrates convolutional neural networks (CNN) for spatial feature extraction, gated recurrent units (GRU) for capturing temporal dependencies, and an attention mechanism to focus on relevant patterns. The model is designed to enhance the precision of cognitive disorder detection by leveraging these sophisticated layers.

**RadarCharts.py:** This script generates radar charts that visualize key features such as Higuchi fractal dimension, Lyapunov exponents, and spectral entropy across AD, CN, and FTD classes. These visualizations highlight significant differences in fractal complexity, neural dynamics, and spectral characteristics, providing insights into the discriminative power of each feature used by the GAEF model for classification.
