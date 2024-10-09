from shapely.geometry import Polygon

def rotated_bbox_iou(bbox1, bbox2):
    """
    회전된 경계 상자의 IoU를 계산합니다.
    
    :param bbox1: [cx, cy, w, h, angle] 형식의 첫 번째 경계 상자
    :param bbox2: [cx, cy, w, h, angle] 형식의 두 번째 경계 상자
    :return: 두 경계 상자의 IoU
    """
    def get_corners(bbox):
        cx, cy, w, h, angle = bbox
        angle = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        half_w, half_h = w / 2, h / 2
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        corners = corners @ rotation_matrix.T
        corners += [cx, cy]
        return corners

    # 두 경계 상자의 코너 좌표를 얻습니다
    corners1 = get_corners(bbox1)
    corners2 = get_corners(bbox2)

    # Shapely Polygon 객체를 생성합니다
    polygon1 = Polygon(corners1)
    polygon2 = Polygon(corners2)

    # 교집합과 합집합의 면적을 계산합니다
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area

    # IoU를 계산합니다
    iou = intersection_area / union_area
    iou = 0 if np.isnan(iou) else iou # 어떤 사유로든 nan인 경우 0처리

    return iou

def f2_with_iou(gt, pr, th=0.01):
    tp_iou = []
    tp = []
    fp = []
    fn = []
    
    for img in list(set(gt['file_name'])):
        gt_img = gt[gt['file_name'] == img][['cx', 'cy', 'width', 'height', 'angle']]
        pr_img = pr[pr['file_name'] == img][['cx', 'cy', 'width', 'height', 'angle']]

        # 해당 GT에 대한 예측이 있는 경우
        if len(pr_img) > 0:
            ious = [rotated_bbox_iou(i, j) for i in gt_img.values for j in pr_img.values]
            ioumat = np.array(ious).reshape(len(gt_img), -1) # gt_dim:0, pr_dim:1
            
            # pr을 iou가 최대인 gt에 할당
            np.argmax(ioumat, axis=0)
            ioumat = ioumat * (ioumat.max(axis=0, keepdims=True) == ioumat)
            
            # TP_IoU / FP / FN
            max_vals = np.amax(ioumat, axis=1)
            tp_iou.extend([i for i in max_vals if i != 0])
            tp.append(sum(max_vals != 0))
            fp.extend([sum(i > th) -1 for i in ioumat if sum(i > th) >= 2])
            fn.append(sum(max_vals == 0))
        # 해당 GT에 대한 예측이 없는 경우 모든 object를 FN에 추가
        else: 
            fn.append(len(gt_img))
    
    tp_iou = np.sum(tp_iou)
    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)

    # precision - 분모가 0이 될 가능성이 있으므로 nan 처리 필요
    precision = tp_iou / (tp + fp)
    precision = 0 if np.isnan(precision) else precision
    # recall
    recall = tp_iou / (tp + fn)

    f2_score = (1 + (2 ** 2)) * (precision * recall) / (((2 ** 2) * precision) + recall)
    
    return f2_score 