# Binaural HRIR Synthesizer

Simple python script to create binaural audio from a mono audio source and a desired trajectory. The binaural effect is generated by convolution of the mono signal with head-related impulse responses (HRIRs) and dynamically updating the HRIR filter according to the given trajectory. The HRIRs used were captured with 1° resolution at 5 different distances using a Neumann KU100 dummy head [1]. HRIRs are interpolated from the dataset through linear weighting to create a continuous binaural effect in the area spanned by the dataset (360° degrees around the listener, within 0.2 to 1.5 meters distance to the listener). The script accepts a sequence of (x, y) coordinates with timestamps as trajectory, where the x-axis represents left to right and the y-axis represents front to back. The listener is sitting at the origin (0, 0). A sound source at (0.2, 0) would thus be located 0.2 meters to your right and a sound source at (-0.2, 0) is 0.2 meters behind you.

## Installation
The following was tested on Ubuntu 24.04 LTS.
Install ffmpeg (required for audio loading):
```
apt-get install ffmpeg
```
Create and activate environment:
```
conda create -n binaural-synthesizer python==3.12
conda activate binaural-synthesizer
```
Install remaining dependencies:
```
pip install -r requirements.txt
```
Download the SOFA near-field dataset from [here](https://zenodo.org/records/4297951/files/NFHRIR_CIRC360_SOFA.zip?download=1) and extract the **.sofa** files to a location of your choice. The folder containing the **.sofa** files is further referred to as the **dataset_path**.

## Usage
### Quick test with circular trajectory
Point **main.py** to an audiofile of your choice:
```
python main.py dataset_path path/to/your/audiofile
```
This will create a second .wav audiofile, with suffix *_binauralized* in the same location as the provided audiofile, containing the binauralized version. The default trajectory is a continuous circular motion around the listener at 0.5 meters distance, one full rotation taking 5 seconds. After 100 seconds, the source remains fixed 0.5 meters in front of the listener.

### Custom trajectory
Simply include your trajectory text file as an argument:
```
python main.py dataset_path path/to/your/audiofile -t path/to/your/trajectory
```

## Trajectory generation
The trajectory is provided as a text file in which each line contains a tuple (x, y, t), formatted as a comma sparated list. The first tuple always has to be at t=0.0 (start time of the audio) and specifies the initial source location. For example, the following trajectory consisting of a single coordinate places the source initially at x=0.5m, y=0.3m for time t=0.0s:
```
0.5,0.3,0.0
```
The source remains at that position until the timestamp of the next coordinate is reached or the input audio ends. Providing a trajectory text file with only a single coordinate is hence equivalent to statically positioning the source in space. The following example moves the source at time t=2.0s to position x=0.55m and y=0.35m:  
```
0.5,0.3,0.0
0.55,0.35,2.0
```
This can of course be extended to arbitrary complexity and is all that is needed to create arbitrary motions around the listener. In **src/trajectoryutils.py** you additionally find a utility to export numpy arrays in the required format. Some important things to consider when you create your own trajectory:
- make sure the trajectory is smooth, avoid sudden large changes in coordinates
- the HRIR dataset covers distances between 0.2 and 1.5 meters to the listener. Anything outside of that rage is clipped to these limits.
- make sure that the first coordinate is at time t=0.0s

## References
- [1] Arend et al., "Measurement and Perceptual Evaluation of a Spherical Near-Field HRTF Set", [http://audiogroup.web.th-koeln.de/FILES/Arend_TMT2016.pdf](http://audiogroup.web.th-koeln.de/FILES/Arend_TMT2016.pdf)