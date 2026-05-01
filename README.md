# Berramdane-DoubleSlit-V14.3
Berramdane Double‑Slit Simulator – Laboratory‑Grade V14.3: Maxwell‑Boltzmann · Slit Thickness · Vector Potential A · auto_calibrate
# Berramdane Double‑Slit Simulator – V14.3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Berramdane-DoubleSlit-V14.3/blob/main/Berramdane_V14_3.ipynb)

**High‑fidelity, interactive simulation of the electron double‑slit experiment**  
— including relativistic de Broglie wavelength, Feynman path integral, Lindblad decoherence, a realistic detector, and three novel advanced features:

- ✨ **Maxwell‑Boltzmann velocity distribution** (physically correct for thermal electron sources)
- ✨ **Slit thickness model** (amplitude transmission, wall phase shift, backscattering)
- ✨ **Aharonov‑Bohm effect via vector potential A** (path‑dependent phase shift, not a crude offset)
- ✨ **Auto‑calibration** – automatically matches the Jönsson fringe spacing (0.18 mm)
- ✨ **Fully vectorised** – 10–50× faster than previous versions

## ⚙️ Requirements

- Python 3.8+
- `numpy`, `matplotlib`, `scipy`, `ipywidgets` (optional, but needed for GUI)
- Jupyter / Colab environment (recommended)

Install dependencies:
```bash
pip install numpy matplotlib scipy ipywidgets pandas
