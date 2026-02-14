# Installation

## Standard (local)

### Local installation with a virtual environment
1) **Prerequisites**: Python 3.10 installed on your system.
2) **Create a virtual environment**:
   ```bash
   python3.10 -m venv venv
   ```
3) **Activate the environment**:
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
4) **Install PyTorch and xformers**:
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
   ```
   Note: this build targets CUDA 12.1. If you are on CPU-only or a different CUDA version,
   install the appropriate PyTorch build from the official selector.
5) **Install OpenMIM**:
   ```bash
   pip install openmim==0.3.9
   ```
6) **Install `mmengine`/`mmcv`/`mmseg` via MIM**:
   ```bash
   mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
   ```
7) **Install `dense_direction`**:
   ```bash
   pip install .
   ```

## Alternative (Docker)

### Docker installation and remote interpreter
1) **Prerequisites**: Docker installed and running.
2) **Build the image or pull it from GHCR**:
    ```bash
    docker build -t dense-direction .
    ```
   
    ```bash
    docker pull ghcr.io/barwojcik/dense_direction:latest
    ```
3) **Configure a remote interpreter (optional)**:
   - In your IDE or editor, add a Docker-based Python interpreter.
   - Select your Docker server and the `dense-direction` image.
   - Set the interpreter path to `/usr/bin/python3.10` (or `python3`).
   - Set path mappings:
     - Local path: project root on your machine.
     - Remote path: `/dense_direction` (the `WORKDIR` in `Dockerfile`).

Your IDE will then use the container as its interpreter while editing files locally.

#### IDE-specific examples
- **VS Code**:
  - Install the `Dev Containers` and `Python` extensions.
  - Use `Dev Containers: Open Folder in Container...` and select this repo.
  - Pick the Python interpreter inside the container if prompted.
- **PyCharm**:
  - Go to `Settings/Preferences` > `Project` > `Python Interpreter`.
  - Add a new interpreter and choose `Docker`, then select the `dense-direction` image.
  - Set the interpreter path to `/usr/bin/python3.10`.
  - Configure path mappings to `/dense_direction` if not set automatically.
