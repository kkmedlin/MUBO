# MUBO
Five code files:
1. main - Before launching the main file, choose the dataset, title, number of runs, and number of steps. 
2. mubo - Before running MUBO, choose either the data (for Gisette, Abalone, Spambase, Connect4) or smallData (for Ionosphere) file.
3. data - Processes four of the five datasets: Gisette, Abolone, Spambase, Connect4. Change lines 267 and 273 so that Gisette's minority dataset uses 10% of what's available while the others use 100% available.
4. smallData - This file processes the fifth dataset; the only small one that is part of this study: Ionosphere. 
5. model - MUBO's convoluted neural network model; developed to match that of state-of-the-art methods SMOTified-GAN, GBO and SSG.
