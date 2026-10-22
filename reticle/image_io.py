"""PNG decoding limits for scanner-generated pathology images."""
from PIL import PngImagePlugin


def configure_image_decoder():
    # Some PathGen images carry ICC profiles of up to 1.7 MB (Pillow's default limit is 1 MiB).
    PngImagePlugin.MAX_TEXT_CHUNK = 16*1024*1024
    PngImagePlugin.MAX_TEXT_MEMORY = 64*1024*1024
