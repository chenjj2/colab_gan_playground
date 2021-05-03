### Design

Workflow for single run:

1. Generate training data. *
2. Specify GAN model (wrap generator and discriminator into one class).
3. Run ```G = Gan() G.train(data)```
	1. where ```config``` contains information such as learning rate, epoch, batch, 
	what-to-print, what-to-visualize.
	
Workflow for multiple runs:

1. Specified true data generator.
2. Specify Generative & Discriminative model (```model class```).
3. Specify Run ```SimGAN(n_sim, generator_truth, gan_model, config)```

*(If we keep the true data generator, 
we may use it to generate testing data. Furthermore, some evaluation procedure may be 
designed to compare trained generative model and testing data to avoid "eyeballing". 
For example, it may be interesting to look at training discriminative loss and testing 
discriminative loss. More discussion in multipel runs below.)

Reference
https://realpython.com/generative-adversarial-networks/#checking-the-samples-generated-by-the-gan_1

