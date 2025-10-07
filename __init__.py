from .nodes.node_S3 import *

NODE_CLASS_MAPPINGS = {
    "Save Image To S3/R2": SaveImageToS3R2,
    "Load Image From S3/R2": LoadImageFromS3R2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image To S3/R2": "💾 Save Your Image to S3/R2",
    "Load Image From S3/R2": "💾 Load Your Image From S3/R2"
}
