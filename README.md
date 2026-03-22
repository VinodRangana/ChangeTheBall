# ChangeTheBall: Vision-Based Cricket Ball Degradation Analysis

An automated system designed to quantify the structural and surface wear of Test cricket balls using an 8-phase image processing pipeline.

## 🏏 The Problem

In professional cricket, the decision to change a ball is often subjective, based on the umpire's visual inspection. This project aims to standardize that process by calculating:

- **Sphericity Error:** How much the ball has deformed from a perfect circle.
- **Thread Fraying Index:** The level of degradation in the seam.
- **Surface Roughness:** Quantifying scuffs and leather loss.

## 🏗️ System Architecture (OOP)

The project is built using Object-Oriented Programming to ensure modularity and scalability.

- **`CricketBall` Class:** A data model that stores the state and metrics of a single ball.
- **`Pipeline` Modules:** Separate classes for Segmentation, Sphericity, and Roughness analysis.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Core Libraries:**
  - `OpenCV`: Image processing and segmentation.
  - `NumPy`: High-performance mathematical operations on pixel arrays.
  - `Matplotlib`: Visualizing degradation trends.

## 📁 Folder Structure

```text
ChangeTheBall/
├── data/raw/        # Original multi-angle images
├── src/             # Modular Python source code
├── notebooks/       # Experimental sandbox
└── main.py          # Entry point for the application

## Segmentation

- **Procedure**
    -set the threshold value to create mask(actually for the background and then inverted) - inRange
        -use manual threshold at first hand
        -willing to use ROI or maching learning models
    -using morphological opening function distiguish object and background 
        -kernel size detection can be automated
    -then use contours to identified largest object in the mask
    -using bitwise and make the segmentation finalized

- **Problems**
    - threshhold function only work for the single channel : use inRange
    - Grass background contain same hsv as object(ball) : change threshhold value, find the largest contours and remove others

- **notes:
    -if background contain same or nearest hsv values its hard to seperate object from that background.it is safe to use controlled environment
