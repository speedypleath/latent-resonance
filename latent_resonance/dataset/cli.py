import argparse

from .processing import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, process_directory


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to spectrogram PNGs for GAN training."
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to save spectrogram PNGs")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate")
    parser.add_argument("--n-fft", type=int, default=N_FFT, help="FFT window size")
    parser.add_argument(
        "--hop-length", type=int, default=HOP_LENGTH, help="Hop length"
    )
    parser.add_argument(
        "--n-mels", type=int, default=N_MELS, help="Number of mel bands"
    )

    args = parser.parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
