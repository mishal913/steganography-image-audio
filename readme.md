# Image and Audio Steganography System

This project implements a steganography system that hides secret messages inside images and audio files.

The application provides an interactive interface using Gradio where users can:

- Embed secret messages inside images
- Hide messages inside audio files
- Detect tampering and recover hidden messages
- Simulate attacks such as noise, cropping, or tampering
- Analyze audio safety regions for embedding

## Technologies Used

- Python
- Gradio
- NumPy
- Pillow
- SoundFile

## Features

Image Steganography
- Heatmap-based embedding
- Error correction coding
- Tamper detection

Audio Steganography
- LSB message embedding
- Randomized embedding positions
- Signal-to-noise evaluation

Audio Safety Analysis
- Activity heatbar
- Segment ranking for safe embedding

## How to Run

1 Install dependencies:

pip install -r requirements.txt

2 Run the application:

python app.py

3 Open the Gradio interface in your browser.

## Project Purpose

This project was developed as part of my learning in Artificial Intelligence and security-related applications of data hiding techniques.

More projects will be added to this repository as I continue exploring AI and machine learning systems.