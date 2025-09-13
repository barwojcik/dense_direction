# Installation

## Standard

### Local Installation with Virtual Environment

1.  **Prerequisites**: Ensure you have Python 3.10 installed on your system.
2.  **Create a Virtual Environment**:
    ```bash
    python3.10 -m venv venv
    ```
3.  **Activate the Virtual Environment**:
    *   On Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
4.  **Install PyTorch and xformers**:
    ```bash
    pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
    ```
    *Note: This specific PyTorch version is compiled for CUDA 12.1. If you do not have a compatible NVIDIA GPU, you may need to install a CPU-only version of PyTorch or a different CUDA version. Refer to the official PyTorch installation guide for alternatives.*
5.  **Install OpenMIM**:
    ```bash
    pip install openmim==0.3.9
    ```
6.  **Install MMDetection and MMSegmentation via MIM**:
    ```bash
    mim install mmengine==0.10.3 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
    ```
7.  **Install Remaining Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Alternative with Docker

### Docker Installation and Remote Interpreter Configuration

1.  **Prerequisites**: Ensure Docker is installed and running on your system.
2.  **Build the Docker Image**:
    Navigate to the root directory of your project (where `Dockerfile` and `requirements.txt` are located) and build the Docker image.
    ```bash
    docker build -t dense-direction .
    ```
3.  **Configure PyCharm Remote Interpreter**:
    *   Open your project in PyCharm.
    *   Go to `File` > `Settings` (or `PyCharm` > `Preferences` on macOS).
    *   Navigate to `Project: <Your Project Name>` > `Python Interpreter`.
    *   Click on the gear icon (⚙️) next to the "Python Interpreter" dropdown and select `Add New Interpreter...`.
    *   In the "Add Python Interpreter" dialog, choose `Docker`.
    *   Select `Server` as `Docker`. If it's not configured, you might need to add a Docker connection.
    *   Under `Image name`, select `Existing image` and choose `dense-direction` (or the name you used when building the image).
    *   Ensure the `Python interpreter path in container` is set to `/usr/bin/python3.10` (or `python3`).
    *   **Path Mappings**: This is crucial for linking your local project files with the container.
        *   Click the `+` button under "Path mappings".
        *   For `Local path`, select the root directory of your project on your local machine.
        *   For `Remote path`, enter `/dense_direction` (this is the `WORKDIR` defined in the Dockerfile).
        *   Click `OK` to apply the changes.

Now, PyCharm will use the Docker container as its Python interpreter, allowing you to run and debug your code within the Docker environment while editing files locally.