# MUBO

Five code files:
1. To launch code, run the MAIN file. Before launching, you can choose the dataset, title, number of runs, and number of steps. It produces resulting metrics and graphics.
2. data - This file processes five of the six datasets: Gisette, Abolone, Spambase, Shuttle, Connect4. Don't forget to change lines 267 and 273 so that Gisette's minority dataset uses 10/% of what's available while others use 100% available.
3. smallData - This file processes sixth dataset; the only small one that is part of this study: Ionosphere. 
4. model - This file houses the typical convoluted neural network model we used for MUBO; developed to match that of GBO and SSG, where were, in turn, developed to match that of SMOTified-GAN
5. mubo - This file details our new method MUBO for classifying imbalanced data. Don't forget to choose data (for Gisette, Abalone, Spambase, Shuttle, Connect4); or smallData (for Ionosphere)