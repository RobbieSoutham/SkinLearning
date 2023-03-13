
import os
import pickle
import sys
from QMCSobolSequence import Generator
from subprocess import Popen, run
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
    BOUNDS = [
        [10e3, 50e3], # Skin YM
        [1e3, 25e3], # Adipose YM
        [0.48, 0.499], # Skin PR
        [0.48, 0.499], # Skin PR
        [10e-12, 10e-10], # Skin perm
        [10e-12, 10e-10, ] # Adipose perm
        ]

    gen = Generator(DIMS, BOUNDS)
    gen.sample(nSamples)
    gen.saveSamples("samples.pkl")

    return gen.samples  

if __name__ == "__main__":
    NSAMPLES, GENERATE, ID = parseArguments()

    
    TTHA = 5e-3
    TTR = 5e-3
    TTHS = 1e-3
    PATH = "SamplingResults/"

    
    if GENERATE:
        samples = getParameters(NSAMPLES)
    else:
        with open("samples.pkl", "rb") as f:
            samples = pickle.load(f)
        
        with open("missingSamples.pkl", "rb") as f:
            idx = pickle.load(f)[ID]

    if os.path.exists(PATH+idx):
        os.remove(PATH+idx)
        os.makedirs(PATH+idx)
    

    params = "[{} {}], [{} {}], [{} {}]".format(*samples[idx])
    params += f", {TTR}, {TTHA}, {TTHS}, '{idx}'"
    run(f"matlab -nodisplay -r \"run_batch({params})\"", shell=True)
    
    run(f"febio4 -i {PATH}{idx}/Cutometer_out1.feb", shell=True)
    run(f"matlab -nodisplay -r \"read_nodal(1, '{PATH}{idx}')\"", shell=True)
    run(f"febio4 -i {PATH}{idx}/Cutometer_out2.feb -silent", shell=True)
    run(f"matlab -nodisplay -r \"read_nodal(2, '{PATH}{idx}')\"", shell=True)

    for file in os.listdir(f"{PATH}{idx}"):
        if file not in ["Disp1.csv", "Disp2.csv"]:
            os.remove(f"{PATH}{idx}/{file}")
