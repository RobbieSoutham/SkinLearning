
import os
import pickle
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
        [10e-12, 10e-10] # Adipose perm
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
    PATH = "SamplingResults2/"

    
    if GENERATE:
        samples = getParameters(NSAMPLES)
    else:
        with open("newSamples.pkl", "rb") as f:
            samples = pickle.load(f)

    #missing = True
    if not (os.path.exists(PATH)):
        os.makedirs(PATH)
    """
    if "Disp2.csv" in os.listdir(f"{PATH}{ID}"):
        missing = False
        print("Already complete, deleting files")
        for file in os.listdir(f"{PATH}{ID}"):
            if file != "Disp1.csv" and file != "Disp2.csv":
                os.remove(f"{PATH}{ID}/{file}")
    else:
        shutil.rmtree(f"{PATH}{ID}")
        os.makedirs(f"{PATH}{ID}")
        print("Not completed, removing dir")
    """
    #if missing:
    #os.mkdir(PATH+str(ID))
    params = "[{} {}], [{} {}], [{} {}]".format(*samples[ID])
    params += f", {TTR}, {TTHA}, {TTHS}, '{ID}'"
    run(f"matlab -nodisplay -r \"run_batch({params})\"", shell=True)

    run(f"febio4 -i {PATH}{ID}/Cutometer_out1.feb", shell=True)
    run(f"matlab -nodisplay -r \"read_nodal(1, '{PATH}{ID}')\"", shell=True)
    run(f"febio4 -i {PATH}{ID}/Cutometer_out2.feb -silent", shell=True)
    run(f"matlab -nodisplay -r \"read_nodal(2, '{PATH}{ID}')\"", shell=True)
    
    print("Removing uneeded files")
    for file in os.listdir(f"{PATH}{ID}"):
        if file != "Disp1.csv" and file != "Disp2.csv":
                os.remove(f"{PATH}{ID}/{file}")

