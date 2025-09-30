import fiftyone as fo
import fiftyone.types as fot

# change this to your dataset root
ROOT = "./dataset"

ds = fo.Dataset.from_dir(
    dataset_dir=ROOT,
    dataset_type=fot.YOLOv5Dataset,  # works for YOLOv5/YOLOv8 label .txt
    split="train",  # use "val" to review val set
    label_field="gt",
)

session = fo.launch_app(ds)  # opens a local GUI in your browser
session.wait()  # keep it open
