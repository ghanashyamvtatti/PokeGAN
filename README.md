# PokeGAN - A tf.keras implementation of Deep Convolutional Generative Adversarial Networks
I created this repository to explore tf.keras and get comfortable with GANs.

## Dataset
819 images of Pokemon from Kaggle [Pokemon Images Dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset/home)
Used utils.py to convert them to jpg
resized them to 64x64
<p align="center">
  <img src="https://c1.staticflickr.com/5/4145/4980656042_684be748b7_b.jpg" alt="Pokemon"/>
  <p>© 2010 Pokémon. © 1995-2010 Nintendo/Creatures Inc./GAME FREAK inc.</p>
</p>

## Results
<p align="center">
    <p>Epoch 37</p>
    <img src="https://c1.staticflickr.com/2/1838/44025833242_4b085c7dec_b.jpg"/>
    <p>Epoch 45</p>
    <img src="https://c1.staticflickr.com/2/1811/44025832872_c9f7cfb94d_b.jpg"/>
</p>

## Setup
1. Download the dataset from [here](https://www.kaggle.com/kvpratama/pokemon-images-dataset/home)
2. Convert images to jpg from png (utils.py could help)
3. Install the dependencies using `pip install -r requirements.txt`
4. Change the dataset_path in pokeGAN.py if necessary
5. Run the program `python pokeGAN.py`

## References
- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
- Goodfellow, Ian. "NIPS 2016 tutorial: Generative adversarial networks." arXiv preprint arXiv:1701.00160 (2016).

## Credits
- [Pokemon Images Dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset/home)
- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
- [Anime-Face-GAN-Keras](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)
- [DCGAN and WGAN implementation on Keras for Bird Generation](https://github.com/Goldesel23/DCGAN-for-Bird-Generation)