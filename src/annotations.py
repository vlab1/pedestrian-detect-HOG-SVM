from config import PATCH_WIDTH

def read_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                img_id = int(parts[0])
                y0 = int(parts[1])
                x0 = int(parts[2])
                y1 = int(parts[3])
                x1 = x0 + PATCH_WIDTH
                annotations.append((img_id, x0, y0, x1, y1))
    return annotations
