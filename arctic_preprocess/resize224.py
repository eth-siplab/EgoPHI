import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg
from config import ARCTIC_DATA_ROOT, IMAGES_ROOT

src_root = os.path.join(ARCTIC_DATA_ROOT, "data", "images")
dst_root = IMAGES_ROOT

target_size = (224, 224)

resize = T.Resize(target_size)
device = torch.device(os.environ.get("EGOPHI_RESIZE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))

do_folders = {"s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"}

for s_folder in sorted(os.listdir(src_root)):
    if not s_folder.startswith('s') or s_folder not in do_folders:
        continue
    print(f"Processing {s_folder}")

    s_path = os.path.join(src_root, s_folder)
    if not os.path.isdir(s_path):
        continue

    for obj_folder in sorted(os.listdir(s_path)):
        obj_path = os.path.join(s_path, obj_folder)
        if not os.path.isdir(obj_path):
            continue

        dst_obj_path = os.path.join(dst_root, s_folder, obj_folder)
        if os.path.exists(dst_obj_path):
            print(f"Skipping {obj_folder} (already exists)")
            continue

        print(f"Processing object folder {obj_folder}")

        for cam_folder in sorted(os.listdir(obj_path)):
            if cam_folder != '0':
                continue

            cam_path = os.path.join(obj_path, cam_folder)
            dst_cam_path = os.path.join(dst_obj_path, cam_folder)
            os.makedirs(dst_cam_path, exist_ok=True)

            for img_file in sorted(os.listdir(cam_path)):
                if not img_file.lower().endswith(".jpg"):
                    continue

                src_img_path = os.path.join(cam_path, img_file)
                dst_img_path = os.path.join(dst_cam_path, img_file)

                img = read_image(src_img_path).float() / 255.0  # CxHxW, float tensor
                img = img.unsqueeze(0).to(device)

                img_resized = resize(img)
                img_resized = (img_resized.squeeze(0) * 255).byte().cpu()

                write_jpeg(img_resized, dst_img_path)

print("Done resizing images on GPU, skipping existing object folders.")
