### License plate project comparison between Variational Auto Encoders (Kingma 2022) and DRAW VAE (Gregor 2015).

This project includes code from https://github.com/ericjang/draw

Licensed under the Apache License 2.0.

you need a 3.7 Python env to run some files, including: 
1. draw_plates_rectangle2.py
2. plot_plates_results.py

usage:
### To build the DRAW model:
1. run draw_plates_rectangle2.py --> uses draw_plate_data_prepared_combined\license_plates_combined_96x32.npy to create the DRAW model of the plates.
2. run plot_plates_results.py --> plots the reconstruction results of the model.
### To build the VAE model:
1. go to the **vae-and-draw** branch and run the **VAEBaseline.ipynb** notebook

We're using data from:
1. https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
2. https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset

Paper of reference for DRAW VAE:
Gregor, K., Danihelka, I., Graves, A., & Wierstra, D. (2015). DRAW: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623. http://arxiv.org/abs/1502.04623

Reference for VAE:
Kingma, D. P., & Welling, M. (2022). Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114. https://arxiv.org/abs/1312.6114
