# DeepLab-V2-Seg-ResNet-101-

Requirements
```bash
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```
Add the weights from this [link](https://drive.google.com/file/d/1sBU-HwPqFvSrDP2eN8ZciL51i8MnZNtt/view?usp=sharing) to the model directory
Pass configuration file path to main
```bash
python3 train.py
```
Currently Supports only training