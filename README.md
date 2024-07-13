# This is a plotly Dash Demo for using the Audio2Vec package for annotation!

# Voice Classification Tool

## Dash Community!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-v2-orange)](https://dash.plotly.com/)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

This tool provides a user-friendly interface for recording, playing, and classifying audio samples using machine learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Models & Data](#models--data)
- [Contributing](#contributing)

## Project Overview

**Purpose:** The Voice Classification Tool enables users to experiment with audio recording and classification using both pre-trained and custom machine learning models. 

**Target Users:**

- **Machine Learning Enthusiasts:**  Explore audio classification and experiment with different models.
- **Researchers:** Use the tool for quick prototyping or data collection.

## Features

- **Audio Recording:** Capture audio samples directly in the browser.
- **Audio Playback:** Listen to recorded samples.
- **Classification:** Analyze audio and display the predicted class.
- **Model Flexibility:** Supports different classification algorithms (e.g., k-nearest neighbors, neural networks).
- **Visualization:** Display audio data (e.g., as a spectrogram). 
- **Basic Authentication:** Protects the tool with a simple username and password. 

## Technologies

- **Dash:** Python framework for building web applications.
- **Dash Bootstrap Components (dbc):** For styling and layout.
- **dash-recording-components:**  Library for audio recording in Dash.
- **audio2vec:** (Assuming this is a custom library) For audio feature extraction.
- **scikit-learn (sklearn):** For machine learning models (k-NN).
- **Other:** NumPy, SoundFile, etc.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

```bash
pip install -r requirements.txt

python app.py
```

Usage
 - Access the App: Open your web browser and navigate to the app's URL.
 - Record Audio: Click the "Record" button and speak or provide audio input.
 - Stop Recording: Click the "Stop Recording" button to end the recording.
 - Play Audio: Click the "Play" button to listen to your recording.
 - View Classification: The app will display the predicted class of the audio.

File Structure
 - app.py: Main application file with Dash layout and callbacks.
 - knn.py: k-Nearest Neighbors classification model.
 - nn.py: Neural network classification model.
 - outofSample.py: (If applicable) Script for handling out-of-sample data or testing.
 - audio2vec.py: (If applicable) Custom library for audio feature extraction.
 - /assets/logoA.png: Logo image.

Models & Data
 - Classification Models:
 - k-Nearest Neighbors (k-NN): A simple and effective algorithm for audio classification.
 - Neural Network: A potentially more powerful model for complex audio patterns.

Contributing
 - Contributions are welcome! Please follow the standard GitHub fork and pull request workflow.
