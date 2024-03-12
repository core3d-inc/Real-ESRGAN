from modal import Image, enter, exit, method, Stub, Volume, web_endpoint
from modal.functions import FunctionCall
from typing import Dict

stub = Stub("upscaler")

image = (
    Image
        .debian_slim(
            python_version="3.8",
        )
        .apt_install(
            "python3-opencv",
        )
        .pip_install(
            "numpy==1.20",
            "torch==1.11.0",
            "torchvision==0.12.0",
            "facexlib==0.2.3",
            "opencv-python==4.6.0.66",
            "Pillow==9.1.1",
            "tqdm==4.64.0",
            "basicsr",
            "gfpgan",
        )
        .copy_local_dir("realesrgan", "/usr/local/lib/python3.8/site-packages/realesrgan")
)

volume = Volume.from_name("upscaler-data", create_if_missing=True)

VOLUME_PATH = "/shared"

@stub.cls(
    container_idle_timeout=30,
    cpu=1.0,
    gpu="t4",
    image=image,
    memory=16384,
    volumes={"/shared": volume}
)
class Model:
    @enter()
    def setup(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import uuid, os

        self.dirpath = os.path.join(VOLUME_PATH, str(uuid.uuid4()))

        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=os.path.join(VOLUME_PATH, "RealESRGAN_x4plus.pth"),
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)

    @exit()
    def cleanup(self):
        import shutil
        shutil.rmtree(self.dirpath)
        volume.commit()

    @method()
    def predict(self, url, scale, tile=None):
        from shutil import copyfileobj
        from urllib.request import urlopen
        import cv2, os, uuid

        os.mkdir(self.dirpath)

        input_filepath = os.path.join(self.dirpath, "%s.png" % uuid.uuid4())
        output_filepath = os.path.join(self.dirpath, "%s.png" % uuid.uuid4())

        with urlopen(url) as in_stream, open(input_filepath, 'wb') as out_file:
            copyfileobj(in_stream, out_file)

        image = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
        output, *_ = self.upsampler.enhance(image, outscale=scale, tile=tile)

        cv2.imwrite(output_filepath, output)

        os.unlink(input_filepath)

        volume.commit()

        return output_filepath


    @web_endpoint(method="POST")
    def queue(self, data: Dict):

        if "scale" not in data:
            raise Exception("scale is required")

        scale = float(data["scale"])

        if scale <= 0 or scale > 16:
            raise Exception("scale must be > 0 and <= 16")

        tile = None
        if "tile" in data:
            tile = int(data["tile"])
            if tile <= 100:
                tile = None

        if "url" not in data:
            raise Exception("url is required")

        url = data["url"]

        call = self.predict.spawn(url, scale, tile)

        return {"id": call.object_id}


    @web_endpoint(method="POST")
    def status(self, data: Dict):
        import fastapi

        if "id" not in data:
            raise Exception("id is required")

        id = data["id"]

        function_call = FunctionCall.from_id(id)

        try:
            result = function_call.get(timeout=0)
        except TimeoutError:
            return fastapi.responses.JSONResponse(content="", status_code=202)

        volume.reload()

        return fastapi.responses.FileResponse(result)


@stub.local_entrypoint()
def main(scale, url):
    output = Model.predict.remote(url, scale)
    print(output)

# @stub.function(
#     container_idle_timeout=30,
#     cpu=1.0,
#     gpu="t4",
#     image=image,
#     memory=16384,
#     volumes={"/shared": volume}
# )
# @web_endpoint(method="POST")
# def predict(data: Dict):
#     pass



# VOLUME_PATH = "./models"

# app = App(
#     name="Real-ESRGAN",
#     runtime=Runtime(
#         cpu=1,
#         memory="16Gi",
#         gpu="T4",
#         image=Image(
#             python_version="python3.8",
#             python_packages=[
#                 "numpy==1.20",
#                 "torch==1.11.0",
#                 "torchvision==0.12.0",
#                 "facexlib==0.2.3",
#                 "opencv-python==4.6.0.66",
#                 "Pillow==9.1.1",
#                 "tqdm==4.64.0",
#                 "basicsr",
#                 "gfpgan",
#             ],
#         ),
#     ),
#     volumes=[Volume(name="Real-ESRGAN", path=VOLUME_PATH)],
# )

# def load_upsampler():
#     model = RRDBNet(
#         num_in_ch=3,
#         num_out_ch=3,
#         num_feat=64,
#         num_block=23,
#         num_grow_ch=32,
#         scale=4)

#     upsampler = RealESRGANer(
#         scale=4,
#         model_path=os.path.join(VOLUME_PATH, "RealESRGAN_x4plus.pth"),
#         model=model,
#         tile=0,
#         tile_pad=10,
#         pre_pad=0,
#         half=True)

#     return upsampler

# autoscaler = QueueDepthAutoscaler(
#     max_tasks_per_replica=30,
#     max_replicas=2,
# )

# @app.task_queue(
#     autoscaler=autoscaler,
#     keep_warm_seconds=30,
#     loader=load_upsampler,
#     outputs=[Output(path="output.png")],
# )
# def predict(**inputs):
#     upsampler = inputs["context"]

#     scale = inputs["scale"]
#     if scale <= 0 or scale > 16:
#         raise Exception("scale must be > 0 and <= 16")

#     tile = 0
#     if "tile" in inputs:
#         tile = inputs["tile"]
#         if tile <= 100 or tile is None:
#             tile = 0

#     image = inputs["image"]

#     if not image:
#         raise Exception("image is required")

#     print("url %s" % image)

#     image, *_ = urllib.request.urlretrieve(image)

#     img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
#     output, *_ = upsampler.enhance(img, outscale=scale, tile=tile)

#     cv2.imwrite("output.png", output)
