from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from modal import Image, enter, exit, method, Secret, Stub, Volume, web_endpoint
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
            "torch==1.11.0+cu115",
            "torchvision==0.12.0+cu115",
            "facexlib==0.2.3",
            "opencv-python==4.6.0.66",
            "Pillow==9.1.1",
            "tqdm==4.64.0",
            "basicsr",
            "gfpgan",
            "boto3[crt]",
            find_links="https://download.pytorch.org/whl/torch_stable.html",
        )
        .copy_local_dir("realesrgan", "/usr/local/lib/python3.8/site-packages/realesrgan")
)

VOLUME_PATH = "/shared"

volume = Volume.from_name("upscaler-data", create_if_missing=True)

auth_scheme = HTTPBearer()

@stub.cls(
    container_idle_timeout=2,
    cpu=1.0,
    gpu="a10g",
    image=image,
    memory=8192,
    secrets=[Secret.from_name("auth-token")],
    timeout=600,
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
        pass

    @method()
    def predict(self, url, scale, tile=None):
        from shutil import copyfileobj
        from urllib.request import urlopen
        import boto3, cv2, os, uuid

        os.mkdir(self.dirpath)

        input_filepath = os.path.join(self.dirpath, "%s.png" % uuid.uuid4())

        with urlopen(url) as in_stream, open(input_filepath, 'wb') as out_file:
            copyfileobj(in_stream, out_file)

        image = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
        output, *_ = self.upsampler.enhance(image, outscale=scale, tile=tile)

        os.unlink(input_filepath)

        output_key = "next/cache/modal/%s.png" % uuid.uuid4()

        s3 = boto3.client("s3")
        image_bytes = cv2.imencode(".png", output)[1].tobytes()
        s3.put_object(Bucket="core3d-production",
                      Key=output_key,
                      Body=image_bytes)

        return "s3://core3d-production/%s" % output_key


@stub.function(
    container_idle_timeout=2,
    secrets=[Secret.from_name("auth-token")],
)
@web_endpoint(method="POST")
def queue(data: Dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import fastapi, os

    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if "scale" not in data:
        raise Exception("scale is required")

    scale = float(data["scale"])

    if scale <= 0 or scale > 16:
        raise Exception("scale must be > 0 and <= 16")

    tile = 512
    if "tile" in data:
        tile = int(data["tile"])
        if tile <= 100:
            tile = 512

    if "url" not in data:
        raise Exception("url is required")

    url = data["url"]

    call = Model().predict.spawn(url, scale, tile)

    return fastapi.responses.JSONResponse(content={"id": call.object_id},
                                          status_code=201)


@stub.function(
    container_idle_timeout=10,
    secrets=[Secret.from_name("auth-token")],
    volumes={"/shared": volume},
)
@web_endpoint(method="POST")
def result(data: Dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import fastapi, os

    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if "id" not in data:
        raise Exception("id is required")

    id = data["id"]

    try:
        result = FunctionCall.from_id(id).get(timeout=0)
        return {"url": result}
    except TimeoutError:
        return fastapi.responses.Response(status_code=202)
    except:
        return fastapi.responses.Response(status_code=410)


@stub.local_entrypoint()
def main(scale, url, tile=None):
    output = Model.predict.remote(url, scale, tile)
    print(output)
