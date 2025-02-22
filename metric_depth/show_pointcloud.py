# show pointcloud (ply) file
import open3d as o3d

pcd = o3d.io.read_point_cloud("outputs/vlcsnap-2025-02-21-22h04m55s001.ply")
o3d.visualization.draw_geometries([pcd])
