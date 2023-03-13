
import os
import pickle
import sys
from QMCSobolSequence import Generator
from subprocess import Popen
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-n", "--nsamples", help="Number of samples", type=int)
parser.add_argument("-i", "--id", type=int,
                    help="ID for the sample")
parser.add_argument("-g", "--generate", default=False,
                    help="To generate or load samples")

args = parser.parse_args()

def parseArguments():

    if args.nsamples:
        while not((args.nsamples != 0) and (args.nsamples & (args.nsamples-1) == 0)):
                args.nsamples = int(input("Ensuring balance properties requires a power of 2"))
#65536
    return args.nsamples, args.generate, args.id

def getParameters(nSamples):
    DIMS = 6
    '''
    BOUNDS = [
        [10e3, 50e3], # Skin YM
        [1e3, 25e3], # Adipose YM
        [0.48, 0.499], # Skin PR
        [0.48, 0.499], # Adipose PR
        [10e-12, 10e-10], # Skin perm
        [10e-12, 10e-10, ] # Adipose perm
        ]
    '''
    with open("../sampleRanges.pkl", "rb") as f:
        BOUNDS = pickle.load(f)
     
    gen = Generator(DIMS, BOUNDS)
    gen.sample(nSamples)
    gen.saveSamples("../newSamples.pkl")

    return gen.samples  

if __name__ == "__main__":
    NSAMPLES, GENERATE, ID = parseArguments()

    
    TTHA = 5e-3
    TTR = 5e-3
    TTHS = 1e-3
    PATH = "SamplingResults/"

    
    if GENERATE:
        samples = getParameters(NSAMPLES)
        raise Exception()
    else:
        with open("samples.pkl", "rb") as f:
            samples = pickle.load(f)

    if not (os.path.exists(PATH)):
        os.makedirs(PATH)
    
    params = "[{} {}], [{} {}], [{} {}]".format(*samples[ID])
    params += f", {TTR}, {TTHA}, {TTHS}, '{ID}'"
    proc = Popen(f"matlab -nodisplay -r \"run_batch({params})\"", shell=True)
    proc.wait()
    
    probe1 = Popen(f"febio -i Cutometer_out1.feb -silent", shell=True)
    probe1.wait()
    Popen(f"matlab -nodisplay -r \"read_nodal(1, {PATH}/{ID})\"", shell=True)
    probe2 = Popen(f"febio -i Cutometer_out2.feb -silent", shell=True)
    Popen(f"matlab -nodisplay -r \"read_nodal(2, {PATH, '/', ID})\"", shell=True)
    
    
    
    print(f"matlab -nodisplay -r \"run_batch({params})\"")