from beam import App, Runtime, Image, Volume, Output, QueueDepthAutoscaler
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2, os
import urllib.request

VOLUME_PATH = "./models"

app = App(
    name="Real-ESRGAN",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_version="python3.8",
            python_packages=[
                "numpy==1.20",
                "torch==1.11.0",
                "torchvision==0.12.0",
                "facexlib==0.2.3",
                "opencv-python==4.6.0.66",
                "Pillow==9.1.1",
                "tqdm==4.64.0",
                # move to commands?
                "basicsr",
                "gfpgan",
            ],
            commands=[
                # "pip install basicsr",
                # "pip install gfpgan",
            ],
        ),
    ),
    volumes=[Volume(name="Real-ESRGAN", path=VOLUME_PATH)],
)

def load_upsampler():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=os.path.join(VOLUME_PATH, "RealESRGAN_x4plus.pth"),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True)

    return upsampler

autoscaler = QueueDepthAutoscaler(
    max_tasks_per_replica=30,
    max_replicas=2,
)

@app.task_queue(
    autoscaler=autoscaler,
    loader=load_upsampler,
    outputs=[Output(path="output.png")],
    # keep_warm_seconds=300,
)
def predict(**inputs):
    upsampler = inputs["context"]
    image, *_ = urllib.request.urlretrieve(inputs["image"])

    scale = inputs["scale"]
    if scale <= 0 or scale > 16:
        raise Exception("scale must be > 0 and <= 16")

    tile = 0
    if "tile" in inputs:
        tile = inputs["tile"]
        if tile <= 100 or tile is None:
            tile = 0

    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    output, *_ = upsampler.enhance(img, outscale=scale, tile=tile)

    cv2.imwrite("output.png", output)
