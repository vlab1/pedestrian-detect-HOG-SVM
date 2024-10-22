import cv2
from annotations import read_annotations
from feature_extraction import extract_patches_and_compute_hog, compute_hog
from svm_training import train_svm, load_svm_model, predict_svm
from utils import generate_random_negative_samples
from metrics import calculate_metrics
from config import TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, TRAIN_ANNOTATIONS_PATH, TEST_ANNOTATIONS_PATH, PATCH_WIDTH, PATCH_HEIGHT

def main():
    while True:
        print("1. Train the Classifier")
        print("2. Test the Classifier")
        print("3. Classify a New Image")
        print("4. Exit")
        choice = input("Select an action: ")

        if choice == '1':
            annotations = read_annotations(TRAIN_ANNOTATIONS_PATH)
            feature_vectors = extract_patches_and_compute_hog(annotations, TRAIN_IMAGES_DIR)

            labels = [1] * len(annotations)
            negative_feature_vectors = generate_random_negative_samples(annotations, TRAIN_IMAGES_DIR, len(annotations))

            feature_vectors.extend(negative_feature_vectors)
            labels.extend([0] * len(negative_feature_vectors))
            train_svm(feature_vectors, labels)

        elif choice == '2':
            annotations = read_annotations(TEST_ANNOTATIONS_PATH)
            feature_vectors = extract_patches_and_compute_hog(annotations, TEST_IMAGES_DIR)

            model = load_svm_model()
            predictions = predict_svm(model, feature_vectors)
        
            detected_boxes = []
            for i, prediction in enumerate(predictions):
                if prediction == 1:
                    img_id = annotations[i][0]
                    x0, y0 = annotations[i][1], annotations[i][2]
                    detected_boxes.append((img_id, x0, y0, x0 + PATCH_WIDTH, y0 + PATCH_HEIGHT))
        
            TP, FP, FN = 0, 0, 0
            for img_id in set(ann[0] for ann in annotations):
                img_predictions = [box for box in detected_boxes if box[0] == img_id]
                img_predictions_formatted = [(box[1], box[2], box[3], box[4]) for box in img_predictions]
                tp, fp, fn = calculate_metrics(img_predictions_formatted, annotations, img_id)
                TP += tp
                FP += fp
                FN += fn
                
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            print(f'Precision: {precision * 100:.2f}%')
            print(f'Recall: {recall * 100:.2f}%')

            if precision >= 0.7 and recall >= 0.7:
                print("Classification meets the required quality (Precision and Recall >= 70%).")
            else:
                print("Classification does not meet the required quality.")    
                
        elif choice == '3':
            image_path = input("Enter the path to the image: ")
            model = load_svm_model()
            image = cv2.imread(image_path)
            if image is None:
                print("Image cannot be uploaded.")
                return

            height, width = image.shape[:2]
            detected_boxes = []
    
            for y in range(0, height - PATCH_HEIGHT, 50):
                for x in range(0, width - PATCH_WIDTH, 50):
                    patch = image[y:y+PATCH_HEIGHT, x:x+PATCH_WIDTH]
                    if patch.shape[0] == PATCH_HEIGHT and patch.shape[1] == PATCH_WIDTH:
                        hog_features = compute_hog(patch)
                        prediction = predict_svm(model, [hog_features])
                        
                        if prediction[0] == 1:
                            print(f"Pedestrian found in the region ({x}, {y}, {x+PATCH_WIDTH}, {y+PATCH_HEIGHT})")
                            detected_boxes.append((x, y, x + PATCH_WIDTH, y + PATCH_HEIGHT))

            for (x0, y0, x1, y1) in detected_boxes:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

            cv2.imshow('Detected Pedestrians', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif choice == '4':
            break
        else:
            print("Wrong choice, try again.")

if __name__ == "__main__":
    main()
