# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from basicsr.archs.rrdbnet_arch import RRDBNet
import os, cv2
import subprocess

subprocess.call(['python', '/src/setup.py', 'develop'])

from realesrgan import RealESRGANer
import tempfile

model_name = 'RealESRGAN_x4plus'
model_path = os.path.join('/root/.cache/realesrgan', model_name + ".pth")

class Predictor(BasePredictor):
    def setup(self):
        pass

    def get_model(self, tile):
        if not hasattr(self, 'model'):
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        if not hasattr(self, 'models_by_tile'):
            self.models_by_tile = {}

        if tile not in self.models_by_tile:
            self.models_by_tile[tile] = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=self.model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=True)

        return self.models_by_tile[tile]

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=4
        ),
        tile: int = Input(
            description=
            'Tile size. Default is 0, that is no tile. When encountering the out-of-GPU-memory issue, please specify, e.g. 256 or 512',
            default=0)
    ) -> Path:
        if tile <= 100 or tile is None:
            tile = 0
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        output, _ = self.get_model(tile).enhance(img, outscale=scale)
        save_path=os.path.join(tempfile.mkdtemp(), "output.png")
        cv2.imwrite(save_path, output)
        return Path(save_path)
