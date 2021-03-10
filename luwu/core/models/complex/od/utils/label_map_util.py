# -*- coding: utf-8 -*-
# @Author       : AaronJny
# @LastEditTime : 2021-03-07
# @FilePath     : /LuWu/luwu/core/models/complex/od/utils/label_map_util.py
# @Desc         :
import glob
import os
import xml.etree.ElementTree as ET

from object_detection.utils.label_map_util import get_label_map_dict


def create_label_map(source_dataset_dir: str, pbtxt_save_path: str = "") -> str:
    """给定一个包含多个xml标注文件的目录，读取标注信息，生成label_map.pbtxt。

    Args:
        source_dataset_dir (str): xml所在的文件夹路径
        pbtxt_save_path (str, optional): label_map.pbtxt的保存路径，如果为空，则不保存. Defaults to ''.

    Returns:
        str: 生成的label_map.pbtxt的内容
    """
    label_dict = {}
    path = os.path.join(source_dataset_dir, "*.xml")
    for xml_file in glob.glob(path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            label = member[0].text
            if label in label_dict:
                continue
            label_dict[label] = len(label_dict) + 1
    items = []
    item_text_format = """
    item {
        id: %s,
        name: '%s'
    }
    """
    for label, xid in label_dict.items():
        items.append(item_text_format % (xid, label))
    content = "\n".join(items)
    if pbtxt_save_path:
        dirname = os.path.dirname(pbtxt_save_path)
        os.makedirs(dirname, exist_ok=True)
        with open(pbtxt_save_path, "w") as f:
            f.write(content)
    return content
