# QuantumMolGAN-PyTorch

This is the PyTorch reimplementation of [Exploring the Advantages of Quantum Generative Adversarial Networks in Generative Chemistry](https://arxiv.org/abs/2210.16823)

## Branches

Different branches contain different experiments of the paper

| Branch Name | Noise Generator | Generator| Discriminator |
| :-----: | :---: | :---: |  :---: |
| main | classical/quantum   | classical   | classical|
| discriminator | classical/quantum   | classical   | classical/quantum|
| generator | classical/quantum   | classical/quantum   | classical/quantum|

## Environment

The environment can be install:

```bash
conda env create -f environment.yml
```

You are able to activate the environment:

```bash
conda activate molgan-pt
```

## Download GDB-9 Dataset

Simply run a bash script in the data directory and the GDB-9 dataset will be downloaded and unzipped automatically together with the required files to compute the NP and SA scores.

```bash
cd data
bash download_dataset.sh
```

The QM9 dataset is located in the data directory as well.

Feel free to use it.

## Data Preprocessing

Simply run the python script within the data direcotry. 

You need to comment or uncomment some lines of code in the main function.

```python
python sparse_molecular_dataset.py
```

## MolGAN and Quantum-GAN

Simply run the following command to train the MolGAN or Quantum-GAN.

```python
python main.py
```

You are able to define the training parameters within the training block of the main function in `main.py`

## Testing Phase

Simply run the same command to test the MolGAN or Quantum-GAN. 

You need to comment the training section and uncomment the testing section in the main function of `main.py`

```python
python main.py
```

## Others

`results` folder stores the log files, trained models, pretrained quantum circuits, and the testing results.

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2210.16823,
  doi = {10.48550/ARXIV.2210.16823},
  
  url = {https://arxiv.org/abs/2210.16823},
  
  author = {Kao, Po-Yu and Yang, Ya-Chu and Chiang, Wei-Yin and Hsiao, Jen-Yueh and Cao, Yudong and Aliper, Alex and Ren, Feng and Aspuru-Guzik, Alan and Zhavoronkov, Alex and Hsieh, Min-Hsiu and Lin, Yen-Chu},
  
  keywords = {Quantum Physics (quant-ph), FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Exploring the Advantages of Quantum Generative Adversarial Networks in Generative Chemistry},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```

## Credits
This repository refers to the following repositories:
 - [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
 - [ZhenyueQin/Implementation-MolGAN-PyTorch](https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch)
 - [jundeli/quantum-gan](https://github.com/jundeli/quantum-gan)
