# FFT Window Functions Demo

Interactive demonstrations of FFT window functions and NMR signal processing.

## Interactive Notebooks

### Marimo Notebook (WASM - Run in Browser)
**[▶️ Open Interactive Notebook](https://varioustoxins.github.io/fft_demos/fft_window_new.wasm.html)**

Modern reactive notebook running entirely in your browser via WebAssembly. No installation needed!

### Jupyter Notebook (Binder)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/varioustoxins/fft_demos/HEAD?urlpath=apps%2Ffft_window.ipynb)

Original Jupyter notebook version. Note: may take some time to start.

## Overview

This notebook demonstrates:
- Sine wave generation and visualization
- FFT transformations
- Window functions for signal processing
- Interactive parameter exploration

## Local Development

**Marimo:**
```bash
pip install marimo numpy scipy matplotlib
marimo edit fft_window_new.py
```

**Jupyter:**
```bash
pip install jupyter numpy scipy matplotlib ipywidgets
jupyter notebook fft_window.ipynb
```
