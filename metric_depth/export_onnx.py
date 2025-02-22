import os

import numpy as np
import onnx
import torch
from onnxconverter_common import float16
from onnxsim import simplify

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    # load models
    max_depth = 80  # for outdoor
    depth_anything = DepthAnythingV2(
        **{
            **{"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "max_depth": max_depth,
        }
    )
    depth_anything.load_state_dict(
        torch.load(
            "models/depth_anything_v2_metric_vkitti_vits.pth", map_location="cpu"
        )
    )
    depth_anything = depth_anything.to("cpu").eval()

    # opencv style bgr numpy array (full hd)
    input_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    input_tensor, (h, w) = depth_anything.image2tensor(input_image, 518)
    input_tensor = input_tensor.to("cpu")

    os.makedirs("export_models", exist_ok=True)

    # export to onnx
    torch.onnx.export(
        depth_anything,
        input_tensor,
        "export_models/depth_anything_v2_metric_vkitti_vits.org.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["output_depth"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "output_depth": {0: "batch_size"},
        },
    )

    # postprocess model
    onnx_model = onnx.load(
        "export_models/depth_anything_v2_metric_vkitti_vits.org.onnx"
    )
    for _ in range(10):
        onnx_model, check = simplify(onnx_model)
        assert check

    onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(onnx_model, "export_models/depth_anything_v2_metric_vkitti_vits.onnx")


if __name__ == "__main__":
    main()
