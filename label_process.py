
from skimage import measure
import numpy as np

def remove_small_block(mask: np.ndarray, thresh_ratio: float):
    mask = mask.astype(np.uint8)
    # 计算图像面积
    area = mask.shape[0] * mask.shape[1]
    # 计算阈值
    threshold = thresh_ratio * area
    # # 查找图像中的轮廓，返回轮廓列表和层次结构
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 遍历轮廓列表
    # for cnt in contours:
    #     # 计算轮廓的面积
    #     area = cv2.contourArea(cnt)
    #     # 如果面积小于阈值，用黑色填充该轮廓
    #     if area < threshold:
    #         cv2.drawContours(mask, [cnt], 0, 0, -1)
    img_label, num = measure.label(mask, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    resMatrix = np.zeros(img_label.shape)
    for i in range(1, len(props)):
        if props[i].area > threshold:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 1
    return resMatrix


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def segment_boxes(h, w, edge):
    boxes = []
    h1 = edge * h
    h2 = (1 - edge) * h
    w1 = edge * w
    w2 = (1 - edge) * w
    box1 = [0, 0, w1, h]
    box2 = [w2, 0, w, h]
    box3 = [w1, 0, w2, h1]
    box4 = [w1, h2, w2, h]
    boxes.append(box1)
    boxes.append(box2)
    boxes.append(box3)
    boxes.append(box4)

    return boxes