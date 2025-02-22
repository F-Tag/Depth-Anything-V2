python depth_to_pointcloud.py --encoder vits --load-from models/depth_anything_v2_metric_vkitti_vits.pth --max-depth 80 --img-path vlcsnap-2025-02-21-22h04m55s001.png --outdir .\outputs
python run.py --encoder vits --load-from models/depth_anything_v2_metric_vkitti_vits.pth --max-depth 80 --img-path vlcsnap-2025-02-21-22h04m55s001.png --outdir outputs --grayscale
python .\show_pointcloud.py  

python run_onnx.py --load-from export_models/depth_anything_v2_metric_vkitti_vits.onnx --img-path vlcsnap-2025-02-21-22h04m55s001.png --outdir outputs --grayscale