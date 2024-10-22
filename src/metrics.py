def calculate_metrics(predictions, annotations, img_id):
    true_annotations = [ann for ann in annotations if ann[0] == img_id]
    
    TP = FP = FN = 0
    
    for pred in predictions:
        pred_x0, pred_y0, pred_x1, pred_y1 = pred

        for ann in true_annotations:
            ann_x0, ann_y0, ann_x1, ann_y1 = ann[1], ann[2], ann[3], ann[4]

            inter_x0 = max(pred_x0, ann_x0)
            inter_y0 = max(pred_y0, ann_y0)
            inter_x1 = min(pred_x1, ann_x1)
            inter_y1 = min(pred_y1, ann_y1)

            if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                intersection_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                pred_area = (pred_x1 - pred_x0) * (pred_y1 - pred_y0)
                ann_area = (ann_x1 - ann_x0) * (ann_y1 - ann_y0)

                if intersection_area > 0.5 * pred_area and intersection_area > 0.5 * ann_area:
                    TP += 1
                    break
        else:
            FP += 1

    FN = len(true_annotations) - TP

    return TP, FP, FN
