import tensorflow as tf

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """計算 IoU, Precision, Recall, F1"""
    y_pred_thresholded = y_pred > threshold
    
    # Mean IoU
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
    iou_metric.update_state(y_pred_thresholded, y_true)
    mean_iou = iou_metric.result().numpy()
    
    # Precision
    prec_metric = tf.keras.metrics.Precision()
    prec_metric.update_state(y_pred_thresholded, y_true)
    precision = prec_metric.result().numpy()
    
    # Recall
    recall_metric = tf.keras.metrics.Recall()
    recall_metric.update_state(y_pred_thresholded, y_true)
    recall = recall_metric.result().numpy()
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        "Mean IoU": mean_iou,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }