
import sys
from QMCSobolSequence import Generator

if __name__ == "__main__":
    DIMS = 6
    BOUNDS = [
        [10e3, 50e3], # Skin YM
        [1e3, 25e3], # Adipose YM
        [0.48, 0.499], # Skin PR
        [0.48, 0.499], # Skin PR
        [10e-12, 10e-10], # Skin perm
        [10e-12, 10e-10, ] # Adipose perm
        ]
    TTHA = 5e-3
    TTR = 5e-3
    TTHS = 1e-3

    NSAMPLES = int(sys.argv[1])
    gen = Generator(DIMS, BOUNDS)
    
    gen.sample(NSAMPLES)
    gen.saveSamples("samples.pkl")

    for i, sample in enumerate(gen.samples):
        #proc = Popen(f"cd Model_res && matlab -nodisplay -r \"run_batch({command})\"", shell=True)
        #procs.append(proc)
        print(sample)
        params = "[{} {}], [{} {}], [{} {}]".format(*sample)
        params += f", {TTR}, {TTHA}, {TTHS}, '{i}'"
        print(f"matlab -nodisplay -r \"run_batch({params})\"")

