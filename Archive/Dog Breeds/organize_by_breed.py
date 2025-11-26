#!/usr/bin/env python3
"""
Script to organize dog breed images into breed-specific folders.
Restructures from flat directory to train/breed/ and test/breed/ structure.
"""

import os
import shutil
import csv
from pathlib import Path
from collections import defaultdict

def organize_images_by_breed(base_dir):
    """
    Organize images in train/ and test/ folders by breed.
    
    Args:
        base_dir: Base directory containing labels.csv, train/, and test/
    """
    base_path = Path(base_dir)
    labels_file = base_path / 'labels.csv'
    train_dir = base_path / 'train'
    test_dir = base_path / 'test'
    
    # Read labels
    print("Reading labels.csv...")
    image_to_breed = {}
    breeds = set()
    
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['id']
            breed = row['breed']
            image_to_breed[image_id] = breed
            breeds.add(breed)
    
    print(f"Found {len(image_to_breed)} labeled images across {len(breeds)} breeds")
    
    # Create breed directories in train/
    print("\nCreating breed directories in train/...")
    for breed in breeds:
        breed_dir = train_dir / breed
        breed_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize training images
    print("\nOrganizing training images by breed...")
    train_images = list(train_dir.glob('*.jpg'))
    organized_count = 0
    skipped_count = 0
    
    for img_path in train_images:
        image_id = img_path.stem  # filename without extension
        
        if image_id in image_to_breed:
            breed = image_to_breed[image_id]
            dest_dir = train_dir / breed
            dest_path = dest_dir / img_path.name
            
            # Move the image
            shutil.move(str(img_path), str(dest_path))
            organized_count += 1
            
            if organized_count % 1000 == 0:
                print(f"  Organized {organized_count} training images...")
        else:
            skipped_count += 1
    
    print(f"Organized {organized_count} training images")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images without labels")
    
    # Handle test images (if any have labels)
    print("\nChecking test images...")
    test_images = list(test_dir.glob('*.jpg'))
    
    if test_images:
        print(f"Found {len(test_images)} test images")
        
        # Create breed directories in test/ for any labeled test images
        test_organized = 0
        test_unlabeled = []
        
        for img_path in test_images:
            image_id = img_path.stem
            
            if image_id in image_to_breed:
                breed = image_to_breed[image_id]
                breed_dir = test_dir / breed
                breed_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = breed_dir / img_path.name
                shutil.move(str(img_path), str(dest_path))
                test_organized += 1
                
                if test_organized % 1000 == 0:
                    print(f"  Organized {test_organized} test images...")
            else:
                test_unlabeled.append(img_path)
        
        print(f"Organized {test_organized} test images with labels")
        
        if test_unlabeled:
            print(f"\nFound {len(test_unlabeled)} unlabeled test images")
            # Create an 'unlabeled' folder for test images without labels
            unlabeled_dir = test_dir / 'unlabeled'
            unlabeled_dir.mkdir(exist_ok=True)
            print("Moving unlabeled test images to test/unlabeled/...")
            
            for img_path in test_unlabeled:
                dest_path = unlabeled_dir / img_path.name
                shutil.move(str(img_path), str(dest_path))
            
            print(f"Moved {len(test_unlabeled)} unlabeled images to test/unlabeled/")
    
    # Print summary
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"Total breeds: {len(breeds)}")
    print(f"Training images organized: {organized_count}")
    print(f"Test images organized: {test_organized if test_images else 0}")
    print("\nDirectory structure:")
    print(f"  train/")
    for breed in sorted(breeds)[:5]:  # Show first 5 breeds
        count = len(list((train_dir / breed).glob('*.jpg')))
        print(f"    {breed}/ ({count} images)")
    if len(breeds) > 5:
        print(f"    ... and {len(breeds) - 5} more breeds")
    
    if test_images and test_organized > 0:
        print(f"  test/")
        test_breeds = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != 'unlabeled']
        for breed in sorted(test_breeds)[:5]:
            count = len(list((test_dir / breed).glob('*.jpg')))
            print(f"    {breed}/ ({count} images)")
        if len(test_breeds) > 5:
            print(f"    ... and {len(test_breeds) - 5} more breeds")

if __name__ == '__main__':
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    print("Dog Breed Image Organizer")
    print("="*60)
    print(f"Working directory: {script_dir}")
    print()
    
    # Confirm before proceeding
    response = input("This will reorganize images into breed folders. Continue? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        organize_images_by_breed(script_dir)
        print("\nDone!")
    else:
        print("Operation cancelled.")

