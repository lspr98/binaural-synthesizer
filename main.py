import os
import copy
import argparse

from src.audioutils import *
from src.trajectoryutils import *
from src.auralizer import Auralizer

DEFAULT_SR = 48000

HRIR_CONFIG = [
    {
        "dist": 0.25,
        "gain": 1.0,
        "path": "HRIR_CIRC360_NF025.sofa"
    },
    {
        "dist": 0.50,
        "gain": 0.33,
        "path": "HRIR_CIRC360_NF050.sofa"
    },
    {
        "dist": 0.75,
        "gain": 0.25,
        "path": "HRIR_CIRC360_NF075.sofa"
    },
    {
        "dist": 1.0,
        "gain": 0.16,
        "path": "HRIR_CIRC360_NF100.sofa"
    },
    {
        "dist": 1.5,
        "gain": 0.095,
        "path": "HRIR_CIRC360_NF150.sofa"
    }
]



def main():

    parser = argparse.ArgumentParser(description='Utility to binauralize a given input audio file.')
    parser.add_argument('ds_path', help='Path to HRIR dataset folder, containing .sofa files')
    parser.add_argument('input_path', help='Path to input audio file')
    parser.add_argument('-t', '--trajectory_path', nargs='?', const='none', help='Path to trajectory text file')
    parser.add_argument('-s', '--samplingrate', nargs='?', const='none', help='Desired audio sampling rate')
    parser.add_argument('--force', action='store_true', help='Allow overwriting existing files')

    args = parser.parse_args()


    # Input parsing
    ds_path = args.ds_path
    input_path = args.input_path
    trajectory_path = args.trajectory_path
    force = args.force
    samplingrate = args.samplingrate

    if trajectory_path is not None:
        assert os.path.exists(trajectory_path), f"Could not find file {trajectory_path}"
        traj = import_trajectory(trajectory_path)
    else:
        # If no trajectory was given, create a cyclic motion as a demo
        traj = create_circle_trajectory(0.5, 5, 100)

    if samplingrate is not None:
        samplingrate = int(samplingrate)
    else:
        samplingrate = DEFAULT_SR

    assert os.path.exists(ds_path), f"Could not find dataset folder {ds_path}"
    assert os.path.exists(input_path), f"Could not find input audio file {input_path}"

    # Construct output filename
    output_dir, input_filename  = os.path.split(input_path)
    output_filename = os.path.splitext(input_filename)[0] + "_binauralized.wav"
    output_path = os.path.join(output_dir, output_filename)

    assert not os.path.exists(output_path) or force, f"File {output_path} already exists. Use with --force to allow overwriting."

    hrir_config = copy.deepcopy(HRIR_CONFIG)
    for itm in hrir_config:
        itm["path"] = os.path.join(ds_path, itm["path"])

    # Load audio as mono with given samplingrate
    y_mono = load_audio(input_path, samplingrate)

    # Convert to stereo by applying HRIRs
    auralizer = Auralizer(hrir_config)
    y_stereo = auralizer.auralize(y_mono, samplingrate, traj)

    

    # Save audio
    save_audio(y_stereo, output_path, samplingrate, force=force)

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()