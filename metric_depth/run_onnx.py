import argparse
import glob
import os
from datetime import datetime

import cv2
import matplotlib
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


class MetricDepthEstimator:
    def __init__(self, onnx_file):
        self.max_depth = 80
        # setup onnx inference session
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        # "trt_int8_enable": True,
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_int8_use_native_calibration_table": False,
                    "trt_timing_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_sparsity_enable": True,
                },
            ),
            "CUDAExecutionProvider",
            ("OpenVINOExecutionProvider", {"device_type": "CPU", "num_of_threads": 1}),
            "CPUExecutionProvider",
        ]
        self.session = onnxruntime.InferenceSession(
            onnx_file,
            providers=providers,
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[1:]

        # initalize session
        self.session.run(
            None, {self.input_name: np.zeros([1] + self.input_shape, dtype=np.float32)}
        )

    @torch.no_grad()
    def forward(self, x):
        # batch, _, h, w = x.shape
        # io_binding = self.session.io_binding()
        # io_binding.bind_cpu_input(self.input_name, x.numpy())
        # io_binding.bind_output(self.output_name)
        # self.session.run_with_iobinding(io_binding)
        # pred = io_binding.get_outputs()[0]

        pred = self.session.run([self.output_name], {self.input_name: x.numpy()})[0]
        return torch.from_numpy(pred)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)

        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]

        return depth.cpu().numpy()

    @torch.no_grad()
    def image2tensor(self, raw_image, input_size=518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        return image, (h, w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 Metric Depth Estimation"
    )

    parser.add_argument("--img-path", type=str)
    parser.add_argument("--outdir", type=str, default="./vis_depth")

    parser.add_argument(
        "--load-from",
        type=str,
        default="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
    )

    parser.add_argument(
        "--save-numpy",
        dest="save_numpy",
        action="store_true",
        help="save the model raw output",
    )
    parser.add_argument(
        "--pred-only",
        dest="pred_only",
        action="store_true",
        help="only display the prediction",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="do not apply colorful palette",
    )

    args = parser.parse_args()

    depth_anything = MetricDepthEstimator(args.load_from)

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, "**/*"), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral")

    for k, filename in enumerate(filenames):
        print(f"Progress {k + 1}/{len(filenames)}: {filename}")

        raw_image = cv2.imread(filename)

        start = datetime.now()
        depth = depth_anything.infer_image(raw_image)
        print(f"Inference time: {datetime.now() - start}")

        fig = plt.figure()
        depth4plot = depth.flatten()
        depth4plot = depth4plot[depth4plot < 35]
        plt.hist(depth4plot, bins=100)
        plt.savefig(
            os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(filename))[0] + "_hist.onnx.png",
            )
        )
        plt.close()

        if args.save_numpy:
            output_path = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(filename))[0]
                + "_raw_depth_meter.npy",
            )
            np.save(output_path, depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(
            args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".onnx.png"
        )
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)
