import os
import tempfile
from pathlib import Path

import cog
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class Predictor(cog.Predictor):

    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    @cog.input("image", type=Path, help="input image")
    @cog.input(
        'model_name',
        type=str,
        default='RealESRGAN_x4plus',
        options=['RealESRGAN_x4plus', 'RealESRNet_x4plus'],
        help='Model names')
    @cog.input('outscale', type=float, min=1, max=8, default=4, help='The final upsampling scale of the image')
    @cog.input('face_enhance', type=bool, default=False, help='Use GFPGAN to enhance face')
    @cog.input('half', type=bool, default=True, help='Use half precision during inference')
    def predict(self, image, model_name, outscale, face_enhance, half):

        tile = 0
        tile_pad = 10
        pre_pad = 0
        alpha_upsampler = 'realesrgan'
        ext = 'auto'

        # model name
        if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        elif model_name in [
                'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
        ]:  # x2 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
            netscale = 2
        elif model_name in [
                'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
        ]:  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4

        # determine model paths
        model_path = os.path.join('weights', model_name + '.pth')

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half)

        if face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)

        extension = os.path.splitext(os.path.basename(str(image)))[1]
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            cv2.imwrite(out_path, output)

            return out_path
