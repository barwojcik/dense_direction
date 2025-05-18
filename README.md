# Dense Direction

### Description
The `dense_direction` is a framework for dense direction estimation of linear objects in images.
It is built on top of popular OpenMMLab's libraries (`mmengine`, `mmcv`, and `mmseg`), and utilizes loss-based 
algorithmic weak-supervision to learn the direction estimation of linear objects from semantic segmentation maps.

### Key Features
- **Loss-Based Weak-Supervision**: Our approach leverages loss functions to guide the learning process, reducing the need for explicit labeling. There is no need for direction labels, only widely available semantic segmentation maps are needed.
- **Integration with Semantic Segmentation**: The framework seamlessly integrates with semantic segmentation techniques, providing the means to train it alongside direction estimation.
- **Extension of OpenMMLab's frameworks**: By building upon MMSegmentation, we can tap into its established library and expertise in semantic segmentation.
