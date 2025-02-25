# Voicebank2DiffSinger

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3111/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Voicebank2DiffSinger** is a tool that automatically generates training datasets for DiffSinger using **SOFA** and **MakeDiffSinger** from UTAU voicebanks.

[日本語READMEはこちら](README.md)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
  - [Using uv (Fast Installation)](#using-uv-fast-installation)
  - [Using pip](#using-pip)
- [Usage](#usage)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This tool analyzes UTAU voicebank data and converts it into a training dataset for DiffSinger. Internally, it uses the **SOFA model** and **MakeDiffSinger** to automatically process and transform audio data.

## Features

- **Voicebank Analysis:** Extracts phoneme and word sequences from UTAU voicebanks.
- **SOFA Model Usage:** Utilizes the Japanese SOFA model for high-precision audio processing.
- **DiffSinger Data Generation:** Generates training datasets for DiffSinger in conjunction with MakeDiffSinger.

## Directory Structure

```
Directory structure:
└── Voicebank2DiffSinger/
    ├── README.md
    ├── README_EN.md
    ├── LICENSE
    ├── pyproject.toml
    ├── requirements.txt
    ├── uv.lock
    ├── .python-version
    └── src/
        ├── g2p.py
        ├── main.py
        ├── utils.py
        ├── MakeDiffSinger/
        ├── SOFA/
        ├── ckpt/
        │   └── .gitkeep
        ├── dictionaries/
        │   └── .gitkeep
        └── outputs/
            └── .gitkeep
```

## Prerequisites

- **OS:** Windows
- **Development Environment:** C++ (Desktop development using Visual Studio), CMake
- **Python:** 3.11 (Tested with version 3.11.11, versions prior to 3.12)

## Installation Instructions

### Using uv (Fast Installation)

1. **Set up uv (optional)**

   Run the following command in PowerShell:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**

   Clone the repository, including submodules, and navigate to the directory:

   ```powershell
   git clone --recursive https://github.com/Lqm1/Voicebank2DiffSinger.git
   cd Voicebank2DiffSinger
   ```

3. **Install required modules**

   ```powershell
   uv sync
   ```

4. **Install Japanese SOFA model**

   Download the [Japanese SOFA model](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2) and place the following files:

   - Place `step.100000.ckpt` in the `src/ckpt` folder.
   - Place `japanese-extension-sofa.txt` in the `src/dictionaries` folder.

### Using pip

1. **Clone the repository**

   Clone the repository, including submodules, and navigate to the directory:

   ```powershell
   git clone --recursive https://github.com/Lqm1/Voicebank2DiffSinger.git
   cd Voicebank2DiffSinger
   ```

2. **Create and activate a virtual environment**

   ```powershell
   python -m venv .venv
   .venv/scripts/activate
   ```

3. **Install required modules**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Install Japanese SOFA model**

   Download the [Japanese SOFA model](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2) and place the following files:

   - Place `step.100000.ckpt` in the `src/ckpt` folder.
   - Place `japanese-extension-sofa.txt` in the `src/dictionaries` folder.

## Usage

1. **Activate the virtual environment (for pip installation)**

   ```powershell
   .venv/scripts/activate
   ```

2. **Run the tool**

   Run the tool by specifying one or more folders containing the voicebank (note that each folder should contain both the voicebank files and a `.txt` file with label information). Example:

   ```powershell
   python src/main.py example/A3 example/A2 example/A4
   ```

## Notes

- **File Placement:**  
  If the Japanese SOFA model files are not correctly placed in `src/ckpt` and `src/dictionaries`, errors will occur during execution.

- **Dependencies:**  
  This project depends on several external packages. If errors occur during installation, check the Python version and the versions of the required packages.

- **Advanced Configuration:**  
  For detailed settings and customization, refer to the comments in the source code and documentation within each directory.

## Contributing

Contributions such as bug reports, feature suggestions, and pull requests are welcome. Please start by using the [Issue](https://github.com/[username]/Voicebank2DiffSinger/issues) tracker.

## License

This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0).

## Contact

For questions or suggestions, please contact [info@lami.zip](mailto:info@lami.zip).
