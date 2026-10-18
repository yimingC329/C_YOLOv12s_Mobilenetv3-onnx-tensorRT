"""VOC XML 标注转 YOLO txt 格式，并按 train/val 划分图片和标签"""
import os
import shutil
import xml.etree.ElementTree as ET


VOC_ROOT = "data/VOC2028"
YOLO_ROOT = "data"

# 类别映射：VOC 类别名 -> YOLO 类别 id
CLASS_MAP = {"hat": 0, "person": 1}


def voc_to_yolo(xml_path, img_w, img_h):
    """解析单个 VOC XML，返回 YOLO 格式的标注行列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # VOC 的 (xmin,ymin,xmax,ymax) 转 YOLO 的 (cx,cy,w,h)，归一化
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 裁剪到 [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_split(split_name):
    """转换指定划分（train/val）的数据"""
    split_file = os.path.join(VOC_ROOT, "ImageSets", "Main", f"{split_name}.txt")
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    img_dst = os.path.join(YOLO_ROOT, "images", split_name)
    lbl_dst = os.path.join(YOLO_ROOT, "labels", split_name)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    converted = 0
    skipped = 0
    for name in names:
        # 查找图片文件（.jpg 或 .JPG）
        img_path = None
        for ext in (".jpg", ".JPG", ".png"):
            candidate = os.path.join(VOC_ROOT, "JPEGImages", name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        xml_path = os.path.join(VOC_ROOT, "Annotations", name + ".xml")
        if not os.path.exists(xml_path):
            skipped += 1
            continue

        # 解析图片尺寸
        tree = ET.parse(xml_path)
        size = tree.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        # 转换标注
        lines = voc_to_yolo(xml_path, img_w, img_h)

        # 复制图片
        shutil.copy2(img_path, os.path.join(img_dst, os.path.basename(img_path)))

        # 写入 YOLO 标签
        lbl_path = os.path.join(lbl_dst, name + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    print(f"{split_name}: 转换 {converted} 张，跳过 {skipped} 张")


if __name__ == "__main__":
    convert_split("train")
    convert_split("val")
    print("转换完成！")
