# MUBO

Five code files:
1. main - To run MUBO, launch the MAIN file. Before launching, choose the dataset, title, number of runs, and number of steps. 
2. mubo - This file details the MUBO method for classifying imbalanced data. Before running, choose either data (for Gisette, Abalone, Spambase, Shuttle, Connect4) or smallData (for Ionosphere).
3. data - This file processes four of the five datasets: Gisette, Abolone, Spambase, Connect4. Don't forget to change lines 267 and 273 so that Gisette's minority dataset uses 10% of what's available while the others use 100% available.
4. smallData - This file processes the fifth dataset; the only small one that is part of this study: Ionosphere. 
5. model - This file houses the typical convoluted neural network model we used for MUBO; developed to match that of GBO and SSG, which were, in turn, developed to match that of SMOTified-GAN.