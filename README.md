# DeepLab-V2-Seg-ResNet-101-

Requirements
```bash
pip install PyYAML==6.0
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
pip install omegaconf
```
Instructions
- Pass configuration file path to main
- Add model weights to model directory from the following [link](https://drive.google.com/file/d/1sBU-HwPqFvSrDP2eN8ZciL51i8MnZNtt/view?usp=sharing)

```bash
python3 train.py
```
Currently Supports only training