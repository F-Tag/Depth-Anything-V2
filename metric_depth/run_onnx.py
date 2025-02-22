import argparse
import glob
import os
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
from torchvision.transforms import Compose

from depth_anything_v2.util.transform import (NormalizeImage, PrepareForNet,
                                              Resize)


class MetricDepthEstimator:
    def __init__(self, onnx_file):
        self.max_depth = 80
        # setup onnx inference session
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    # "trt_int8_enable": True,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_int8_use_native_calibration_table": False,
                    "trt_timing_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_sparsity_enable": True,
                    "trt_engine_cache_path": "./.cache",
                    "trt_timing_cache_path": "./.cache",
                    # "trt_ep_context_file_path ": "./cache",
                    "trt_dla_enable": True,
                },
            ),
            "CUDAExecutionProvider",
            ("OpenVINOExecutionProvider", {"device_type": "CPU", "num_of_threads": 1}),
            "CPUExecutionProvider",
        ]
        # providers = providers[1:] # unuse TensorrtExecutionProvider
        self.session = onnxruntime.InferenceSession(
            onnx_file,
            providers=providers,
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape[1:]  # (h, w)

        # output tensor
        device = (
            "cuda"
            if torch.cuda.is_available()
            and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            else "cpu"
        )
        self.output_tensor = torch.empty(
            (1, self.output_shape[0], self.output_shape[1]),
            dtype=torch.float32,
            device=device,
        ).contiguous()

        # initalize session
        # opencv style bgr numpy array
        dummy_input = np.zeros(
            (self.output_shape[0], self.output_shape[1], 3), dtype=np.uint8
        )
        self.infer_image(dummy_input)

    @torch.no_grad()
    def forward(self, x):
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input(self.input_name, x.numpy())

        # clear output
        self.output_tensor.zero_()

        # set up io_binding
        io_binding.bind_output(
            self.output_name,
            device_type=self.output_tensor.device.type,
            device_id=self.output_tensor.device.index or 0,
            element_type=np.float32,
            shape=self.output_tensor.shape,
            buffer_ptr=self.output_tensor.data_ptr(),
        )
        self.session.run_with_iobinding(io_binding)
        return self.output_tensor

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)
        depth = depth.squeeze(0)

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

    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="benchmark mode",
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
        count = 100 if args.benchmark else 1
        for _ in range(count):
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
            depth = cv2.resize(depth, (raw_image.shape[1], raw_image.shape[0]))
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)
