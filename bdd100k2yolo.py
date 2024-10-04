import json
import os
import shutil
from tqdm import tqdm
from collections import Counter

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def update_category_mapping(json_data):
    categories = set()
    for item in json_data:
        for label in item['labels']:
            if 'category' in label:
                categories.add(label['category'])
    
    return {category: idx for idx, category in enumerate(sorted(categories))}

def bdd100k_to_yolov8(input_json, output_dir, image_src_dir, image_dst_dir, category_mapping):
    data = load_json(input_json)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dst_dir, exist_ok=True)

    categories_in_json = Counter()
    processed_images = set()
    missing_images = []

    for item in tqdm(data, desc="Processing JSON data"):
        image_file = item['name']
        processed_images.add(image_file)
        output_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + '.txt')
        
        # Copy image file
        src_image_path = os.path.join(image_src_dir, image_file)
        dst_image_path = os.path.join(image_dst_dir, image_file)
        
        try:
            shutil.copy2(src_image_path, dst_image_path)
        except FileNotFoundError:
            missing_images.append(image_file)
            continue  # Skip processing this item if the image is missing

        with open(output_file, 'w') as f:
            for label in item['labels']:
                if 'poly2d' in label:
                    category = label['category']
                    categories_in_json[category] += 1
                    if category not in category_mapping:
                        print(f"Warning: Category '{category}' found in JSON but not in category_mapping")
                        continue
                    category_id = category_mapping[category]
                    poly = label['poly2d'][0]['vertices']
                    
                    if len(poly) < 3:
                        continue
                    
                    poly_norm = [[x/1280, y/720] for x, y in poly]
                    poly_flat = [coord for point in poly_norm for coord in point]
                    poly_str = ' '.join(map(lambda x: f"{x:.6f}", poly_flat))
                    
                    f.write(f"{category_id} {poly_str}\n")

    # Print category statistics
    print("\nCategory statistics:")
    for category, count in categories_in_json.items():
        print(f"{category}: {count} instances")

    # Print missing images
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images were not found:")
        for img in missing_images[:10]:
            print(f"  {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more.")

    return len(processed_images), len(missing_images), dict(categories_in_json)

def process_dataset(base_dir, output_base_dir, split, image_src_dir):
    input_json = os.path.join(base_dir, 'labels', 'sem_seg', 'polygons', f'sem_seg_{split}.json')
    output_dir = os.path.join(output_base_dir, 'labels', split)
    image_dst_dir = os.path.join(output_base_dir, 'images', split)
    
    # Load JSON data and update category mapping
    json_data = load_json(input_json)
    category_mapping = update_category_mapping(json_data)
    
    print(f"\nUpdated category mapping for {split} dataset:")
    for category, idx in category_mapping.items():
        print(f"  {category}: {idx}")
    
    initial_json_entries = len(json_data)
    
    print(f"\nBefore processing {split} dataset:")
    print(f"Number of entries in {split} JSON file: {initial_json_entries}")
    
    processed_count, missing_count, category_instances = bdd100k_to_yolov8(input_json, output_dir, image_src_dir, image_dst_dir, category_mapping)
    
    print(f"\nAfter processing {split} dataset:")
    print(f"Number of images processed and copied to YOLO {split} folder: {processed_count}")
    print(f"Number of label files created in YOLO {split} folder: {processed_count}")
    print(f"Number of missing images: {missing_count}")
    
    print(f"\nAll categories and instances found in {split} dataset:")
    for category in sorted(category_mapping.keys()):
        instances = category_instances.get(category, 0)
        print(f"  {category}: {instances} instances")
    
    return processed_count, missing_count, initial_json_entries, category_mapping, category_instances

if __name__ == '__main__':
    base_dir = r'D:\DataSet\BDD100K\bdd100k_sem_seg_labels_trainval\bdd100k'
    output_base_dir = r'D:\DataSet\BDD100K\BDD100k_YOLO'
    train_images_src_dir = r'D:\DataSet\BDD100K\100k_images_train\bdd100k\images\100k\train'
    val_images_src_dir = r'D:\DataSet\BDD100K\100k_images_val\bdd100k\images\100k\val'
    
    train_results = process_dataset(base_dir, output_base_dir, 'train', train_images_src_dir)
    val_results = process_dataset(base_dir, output_base_dir, 'val', val_images_src_dir)

    print("\nSummary:")
    print(f"Train dataset:")
    print(f"  JSON entries: {train_results[2]}")
    print(f"  Processed images: {train_results[0]}")
    print(f"  Missing images: {train_results[1]}")
    
    print(f"\nValidation dataset:")
    print(f"  JSON entries: {val_results[2]}")
    print(f"  Processed images: {val_results[0]}")
    print(f"  Missing images: {val_results[1]}")

    # Use the category mapping from the train dataset for the data.yaml file
    final_category_mapping = train_results[3]

    print("\nAll categories and total instances in the dataset:")
    for category in sorted(final_category_mapping.keys()):
        train_instances = train_results[4].get(category, 0)
        val_instances = val_results[4].get(category, 0)
        total_instances = train_instances + val_instances
        print(f"  {category}: {total_instances} instances (Train: {train_instances}, Val: {val_instances})")

    # Generate data.yaml file
    data_yaml_content = f"""
train: {os.path.join(output_base_dir, 'images', 'train')}
val: {os.path.join(output_base_dir, 'images', 'val')}

nc: {len(final_category_mapping)}
names: {list(final_category_mapping.keys())}
"""

    with open(os.path.join(output_base_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content)

    print("\nConversion complete. Images copied and data.yaml file has been generated.")