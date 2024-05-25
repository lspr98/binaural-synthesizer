import os
import math
import sofa
import librosa
import numpy as np

from tqdm import tqdm

from .mathutils import *


MAX_CHUNK_SIZE = 1024*4

class Auralizer:
    
    def __init__(self, sofa_file_array) -> None:
        """
    
        Create an Auralizer object to convert mono to binaural audio.

        Parameters
        ----------
        sofa_file_array: list
            List of dictionaries, each containing the path to the individual sofa file along with
            the associated distance and gain
    
        """


        # Clipping for HRIR interpolation
        self.clipping_tol = 1e-4

        # Kernel standard deviation for loudness gain interpolation
        self.sig = 1.0
        
        # Create array of HRIRs from provided file array
        self.ds_arr = []
        self.dist_arr = []
        self.gain_arr = []
        for ds in sofa_file_array:

            curr_dist = ds["dist"]
            curr_path = ds["path"]
            curr_gain = ds["gain"]

            assert os.path.exists(curr_path), f"SOFA dataset file missing: {curr_path}"

            self.ds_arr.append(sofa.Database.open(curr_path))
            self.dist_arr.append(curr_dist)
            self.gain_arr.append(curr_gain)

        self.dist_arr = np.array(self.dist_arr)
        self.gain_arr = np.array(self.gain_arr)

        # Sort dataset by distance
        sort_idx = np.argsort(self.dist_arr)
        self.dist_arr = np.sort(self.dist_arr)
        self.ds_arr = [self.ds_arr[i] for i in sort_idx]

        # Create regression model for audio gain interpolation
        self.weights = self._fit_gain_model()

        # Get sampling rate of HRIRs
        self.sr_H = self.ds_arr[0].Data.SamplingRate.get_values()[0]


    def _fit_gain_model(self):
        """
    
            Performs a simple kernel regression to generate a continuous audio gain model
        
        """

        # Create grammian by evaluating the kernel on the given distances
        X = np.zeros((len(self.dist_arr), len(self.dist_arr)))
        for i, dist in enumerate(self.dist_arr):
            X[:, i] = gaussian(self.dist_arr - dist, self.sig)

        # Get weights from closed form solution by inverting the grammian
        weights = np.linalg.inv(X).dot(self.gain_arr)

        return weights
    

    def _get_interpolated_gain(self, distance):
        """
    
            Evaluate the regression model to get the audio gain associated with the given distance.

            Parameters
            ----------
            distance: float
                Distance to listener in meters

            Returns
            -------
            gain: float
                The audio gain for the given distance, clipped to the range [0, 1]
        
        """

        # Clip distance to range covered by dataset
        distance = np.clip(distance, min(self.dist_arr), max(self.dist_arr))

        # Calculate grammian
        X = np.zeros((1, len(self.dist_arr)))
        for i, dist in enumerate(self.dist_arr):
            X[:, i] = gaussian(distance - dist, self.sig)

        # Evaluate kernel model and clip gain to [0, 1] for safety
        gain = X.dot(self.weights).clip(0, 1)[0]

        return gain
    

    def _convolve_segment(self, h_x, s_x, t_x, idx_start, win_length):
        """
    
            Convolves a source signal s_x with filter h_x and stores the result in target t_x.
            The convolution is performed only over a window defined by the start index idx_start
            and a window size win_length 

            Parameters
            ----------
            h_x: (N, ) np.array
                Filter signal

            s_x: (N, ) np.array
                Source signal

            t_x: (N, ) np.array
                Target signal

            idx_start: int
                Start index for convolution

            win_length: int
                Convolution window length

            Returns
            -------
            None (convolution output is directy written to t_x)
        
        """

        filter_length = h_x.shape[0]
        for i in range(idx_start, idx_start+win_length):
            t_x[i] = np.dot(h_x, np.flip(s_x[i:i+filter_length]))


    def _get_interpolated_hrir(self, angle, distance):
        """
    
            Creates an interpolated head-related impulse response (HRIR) based on a given angle and distance.

            Parameters
            ----------
            angle: int
                Counter-clockwise source angle in degrees

            distance: float
                Source distance in meters

            Returns
            -------
            H_x: (N, 2) np.array
                HRIR for left and right audio channel
        
        """

        # Sanity checking inputs
        assert angle >= 0 and angle <= 359, f"Angle must be in range 0 to 356. Given angle: {angle}"
        assert distance > 0, f"Distance must be > 0. Given distance: {distance}"
        
        if distance < (np.min(self.dist_arr) + self.clipping_tol):
            # Return non-interpolated HRIR from dataset with smallest distance
            H_left = self.ds_arr[0].Data.IR.get_values(indices={"M": angle, "R": 0, "E": 0})
            H_right = self.ds_arr[0].Data.IR.get_values(indices={"M": angle, "R": 1, "E": 0})
        elif distance > (np.max(self.dist_arr) - self.clipping_tol):
            # Return non-interpolated HRIR from dataset with largest distance
            H_left = self.ds_arr[-1].Data.IR.get_values(indices={"M": angle, "R": 0, "E": 0})
            H_right = self.ds_arr[-1].Data.IR.get_values(indices={"M": angle, "R": 1, "E": 0})
        else:
            idx = np.argmin(np.abs(self.dist_arr - distance))
            if self.dist_arr[idx] > distance:
                # nearest distance is larger than current distance
                # second interpolation HRIR is previous element
                idx_lower = idx-1
                idx_upper = idx
                
            else:
                # second interpolation HRIR is next element
                idx_lower = idx
                idx_upper = idx+1
                
            # Calculate scaling weights
            diff = self.dist_arr[idx_upper] - self.dist_arr[idx_lower]
            scaleUpper = (distance - self.dist_arr[idx_lower])/diff
            scaleLower = 1 - scaleUpper
            
            # Get upper and lower impulse responses
            H_left_lower = self.ds_arr[idx_lower].Data.IR.get_values(indices={"M": angle, "R": 0, "E": 0})
            H_right_lower = self.ds_arr[idx_lower].Data.IR.get_values(indices={"M": angle, "R": 1, "E": 0})
            H_left_upper = self.ds_arr[idx_upper].Data.IR.get_values(indices={"M": angle, "R": 0, "E": 0})
            H_right_upper = self.ds_arr[idx_upper].Data.IR.get_values(indices={"M": angle, "R": 1, "E": 0})
            
            # Interpolate by linear weighting
            H_left = H_left_lower*scaleLower + H_left_upper*scaleUpper
            H_right = H_right_lower*scaleLower + H_right_upper*scaleUpper
            
        H_x = np.array([H_left, H_right])

        # Apply loudness gain
        gain = self._get_interpolated_gain(distance)
        H_x *= gain

        return H_x
    

    def _create_chunks(self, n_source, sr, trajectory):
        """
    
            Splits the convolution into chunks of maximum size MAX_CHUNKS

            Parameters
            ----------
            n_source: int
                Number of samples in source audio

            sr: int
                Samplingrate of source audio

            trajectory: (N, 3) np.array
                Trajectory for given audio source

            Returns
            -------
            chunks: list of tuples (coord_id, distance, angle, idx_start, win_length)
                List of chunks that need to be processed for the full audio file.
        
        """
        chunks = []
        for i in range(trajectory.shape[1]):
            x, y, t = trajectory[:, i]

            # Get polar coordinates of current trajectory
            d, theta = get_polar_coords_ccw(x, y)

            # Determine convolution window
            if i == trajectory.shape[1]-1:
                idx_end = n_source
            else:
                idx_end = int(trajectory[2, i+1]*sr)

            idx_start = int(t*sr)
            win_length = min(idx_end, n_source) - idx_start

            # Check if we reached the end of the audio file
            if win_length < 1:
                break

            # Split convolution into chunks for better process tracking
            n_sub_chunks = math.ceil(win_length/MAX_CHUNK_SIZE)
            for j in range(n_sub_chunks-1):
                chunks.append([
                    i, 
                    d, 
                    theta, 
                    idx_start+j*MAX_CHUNK_SIZE, 
                    MAX_CHUNK_SIZE])

            # Dont forget remaining chunk
            chunks.append([
                i, 
                d, 
                theta, 
                idx_start+(n_sub_chunks-1)*MAX_CHUNK_SIZE, 
                win_length - (n_sub_chunks-1)*MAX_CHUNK_SIZE])

        return chunks


    # Creates a virtual stereo sound from a mono sound given angle and distance
    def auralize(self, y_mono, sr, trajectory):
        """
    
            Creates a binaural audio from a mono source and a source trajectory.

            Parameters
            ----------
            y_mono: (N, ) np.array
                Input mono audio

            sr: int
                Samplingrate of y_mono

            trajectory: (N, 3) np.array
                Trajectory for given audio source

            Returns
            -------
            y_stereo: (N, 2) np.array
                Binauralized stereo audio
        
        """

        n_source = y_mono.shape[0]
        y_left = np.zeros((n_source, ), dtype=np.float32)
        y_right = np.zeros((n_source, ), dtype=np.float32)

        # Pad source audio with zeros for convolution
        n_pad = self._get_interpolated_hrir(0, 1.0).shape[1]
        y_mono_pad = np.pad(y_mono, (0, n_pad), 'constant').astype(np.float32)

        # Create convolution chunks
        chunks = self._create_chunks(n_source, sr, trajectory)

        prev_id = -1

        # Process all chunks
        for i in tqdm(range(len(chunks))):
            coord_id, d, theta, idx_start, win_length  = chunks[i]

            if coord_id != prev_id:
                # HRIR update required
                h = self._get_interpolated_hrir(theta, d)

                if sr != self.sr_H:
                    h = librosa.resample(h, self.sr_H, sr)

                    
            # Convolve HRIR of left and right channel with mono audio to create stereo sound.
            self._convolve_segment(h[0, :], y_mono_pad, y_left, idx_start, win_length)
            self._convolve_segment(h[1, :], y_mono_pad, y_right, idx_start, win_length)

        return np.array([y_left, y_right])