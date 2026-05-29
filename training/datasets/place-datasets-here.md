### Dataset Creation

Create a folder, in it, place your audio and caption files. **They must be WAV and TXT format respectively.**

![](https://i.imgur.com/AlDlqBI.png)

## Recommended workflow

1. Create a new dataset folder:
   ```bash
   python training/prepare_dataset.py init my_dataset
   ```
2. Place audio files and matching caption files in `training/datasets/my_dataset/`.
   - `segment_000.wav`
   - `segment_000.txt`
3. Validate the dataset:
   ```bash
   python training/prepare_dataset.py validate training/datasets/my_dataset
   ```
4. If you have audio in another format, convert it first:
   ```bash
   python training/prepare_dataset.py convert raw_audio_folder my_dataset
   ```

## Dataset rules

- Audio files must be WAV (or convertible to WAV) and at least 30 seconds long.
- Each audio file must have a corresponding text file with the same base name.
- The folder should live under `training/datasets/`.
