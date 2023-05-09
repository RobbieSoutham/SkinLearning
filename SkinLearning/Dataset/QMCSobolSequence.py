import math
import numpy as np
from scipy.stats import qmc
import pickle

class Generator():
    """
        Generates Sobol samples using the QMC module and the Sobol engine.

            Parameters:
                dims (int):
                    The number of dimensions to sample.
                bounds (list):
                    The bounds of each dimension.
    """
    def __init__(this, dims, bounds, init=None):
        this.sampler = qmc.Sobol(dims)
        this.bounds = np.array(bounds)
        this.nSamples = 0
        
        if init == None:   
            this.samples = []
        else:
            this.samples = init

    """
        Obtain "n" samples.
    """
    def sample(this, n):
        samples = this.sampler.random_base2(int(math.log(n, 2)))
        samples = qmc.scale(samples, this.bounds[:, 0], this.bounds[:, 1])
        if this.nSamples == 0:
            this.samples = samples
        else:
            this.samples.append(samples)
        this.nSamples += n
        return samples
    
    """
        Save the samples to the given filename.
    """
    def save_samples(this, fname):
        with open(f"{fname}", "wb") as f:
            pickle.dump(this.samples, f)

    """
        Load samples from a given file name.
    """     
    def load_samples(this, fname):
        with open(f"{fname}", "rb") as f:
            this.samples = pickle.load(f)
        
        n = 0
        for s in this.samples:
            n += len(s)
            
        this.nSamples = n
        
        # Reset the sampler and fast forward the number of points sampled
        this.sampler.reset()
        this.sampler.fast_forward(n)