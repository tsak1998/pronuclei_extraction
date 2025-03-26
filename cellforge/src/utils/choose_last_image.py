from pathlib import Path
from PIL import Image

datapath = Path("/home/tsakalis/ntua/phd/cellforge/cellforge/data")
save_path = datapath / "last_frames"

if __name__ == "__main__":

    timelapse_folders = (datapath / "raw_timelapses").glob("*")
    for timelapse_f in timelapse_folders:

        last_frame_pth = max(timelapse_f.glob("*.jpg"), key=lambda x: int(x.stem.split("_")[0]))
        image = Image.open(
            last_frame_pth
        )
        
        image.save(save_path/f"{last_frame_pth.parents[0].name}_{last_frame_pth.stem}.jpg")
