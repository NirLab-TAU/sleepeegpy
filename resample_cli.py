import argparse
from pipeline.pipeline import Pipe


parser = argparse.ArgumentParser()
parser.add_argument(
    'file', 
    help='path to an eeg data file', 
    type=str)
parser.add_argument(
    '-o', '--output', 
    help='output directory, otherwise parent of file', 
    type=str)
parser.add_argument(
    '-sf', '--frequency', 
    help='desired sampling frequency', 
    type=int, 
    default=250)
parser.add_argument(
    '-s', '--subject', 
    help='subject code', 
    type=str)
args = parser.parse_args()

pipe = Pipe(path_to_mff=args.file, output_directory=args.output, subject_code=args.subject)
pipe.resample(args.frequency, save=True)