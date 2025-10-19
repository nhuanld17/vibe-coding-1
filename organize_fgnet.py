# organize_fgnet.py

import os
import shutil
from pathlib import Path
from collections import defaultdict
import re
import random

class FGNETOrganizer:
    """Tự động phân loại FG-NET dataset"""
    
    def __init__(self, source_dir: str, output_dir: str = "./datasets/FGNET_organized"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_filename(self, filename: str):
        """
        Parse FG-NET filename
        
        Args:
            filename: e.g., "001A02.JPG"
            
        Returns:
            dict: {
                'person_id': '001',
                'version': 'A',
                'age': 2,
                'original_name': '001A02.JPG'
            }
        """
        # Remove extension
        name = filename.replace('.JPG', '').replace('.jpg', '')
        
        # Parse pattern: 001A02
        match = re.match(r'(\d{3})([AB])(\d{2})', name)
        
        if not match:
            print(f"WARNING: Cannot parse: {filename}")
            return None
        
        person_id, version, age = match.groups()
        
        return {
            'person_id': person_id,
            'version': version,
            'age': int(age),
            'original_name': filename
        }
    
    def organize_by_person(self):
        """
        Tổ chức theo person ID
        
        Output structure:
        FGNET_organized/
        ├── person_001/
        │   ├── age_02.jpg
        │   ├── age_07.jpg
        │   ├── age_15.jpg
        │   └── age_25.jpg
        ├── person_002/
        │   └── ...
        └── ...
        """
        print("Organizing FG-NET by person ID...")
        
        # Get all image files
        image_files = list(self.source_dir.glob("*.JPG")) + \
                     list(self.source_dir.glob("*.jpg"))
        
        print(f"Found {len(image_files)} images")
        
        # Group by person
        persons = defaultdict(list)
        
        for img_file in image_files:
            parsed = self.parse_filename(img_file.name)
            
            if parsed is None:
                continue
            
            persons[parsed['person_id']].append({
                'path': img_file,
                'age': parsed['age'],
                'version': parsed['version']
            })
        
        if len(persons) == 0:
            print("No images parsed successfully!")
            return {}
        
        # Create organized structure
        for person_id, images in persons.items():
            person_dir = self.output_dir / f"person_{person_id}"
            person_dir.mkdir(exist_ok=True)
            
            # Sort by age
            images = sorted(images, key=lambda x: x['age'])
            
            # Copy files with meaningful names
            for img_info in images:
                new_name = f"age_{img_info['age']:02d}.jpg"
                dest_path = person_dir / new_name
                
                shutil.copy2(img_info['path'], dest_path)
            
            print(f"OK Person {person_id}: {len(images)} images "
                  f"(ages: {[img['age'] for img in images]})")
        
        print(f"\nOrganized {len(persons)} persons!")
        print(f"Output: {self.output_dir}")
        
        return persons
    
    def create_test_pairs(self, min_age_gap: int = 10):
        """
        Tạo file test pairs
        
        Format:
        person_001/age_02.jpg,person_001/age_15.jpg,1,13  # same person, gap 13
        person_001/age_07.jpg,person_002/age_07.jpg,0,0   # different, gap 0
        """
        print(f"\nCreating test pairs (min gap={min_age_gap} years)...")
        
        persons = {}
        
        # Load organized structure
        for person_dir in self.output_dir.glob("person_*"):
            person_id = person_dir.name.split('_')[1]
            images = []
            
            for img_file in sorted(person_dir.glob("*.jpg")):
                age = int(img_file.stem.split('_')[1])
                images.append({
                    'path': str(img_file.relative_to(self.output_dir)),
                    'age': age
                })
            
            persons[person_id] = images
        
        # Generate same-person pairs
        same_pairs = []
        for person_id, images in persons.items():
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    age_gap = images[j]['age'] - images[i]['age']
                    
                    if age_gap >= min_age_gap:
                        same_pairs.append({
                            'img1': images[i]['path'],
                            'img2': images[j]['path'],
                            'label': 1,
                            'age_gap': age_gap
                        })
        
        # Generate different-person pairs (same number as same-pairs)
        diff_pairs = []
        person_ids = list(persons.keys())
        
        random.seed(42)
        
        while len(diff_pairs) < len(same_pairs):
            # Pick 2 random different persons
            p1, p2 = random.sample(person_ids, 2)
            
            # Pick random images
            img1 = random.choice(persons[p1])
            img2 = random.choice(persons[p2])
            
            diff_pairs.append({
                'img1': img1['path'],
                'img2': img2['path'],
                'label': 0,
                'age_gap': 0
            })
        
        # Combine and save
        all_pairs = same_pairs + diff_pairs
        random.shuffle(all_pairs)
        
        pairs_file = self.output_dir / "test_pairs.txt"
        
        with open(pairs_file, 'w') as f:
            f.write("img1,img2,label,age_gap\n")
            for pair in all_pairs:
                f.write(f"{pair['img1']},{pair['img2']},{pair['label']},{pair['age_gap']}\n")
        
        print(f"OK Created {len(same_pairs)} same-person pairs")
        print(f"OK Created {len(diff_pairs)} different-person pairs")
        print(f"Saved to: {pairs_file}")
        
        return pairs_file
    
    def print_statistics(self):
        """In thống kê dataset"""
        print("\n" + "="*60)
        print("FG-NET DATASET STATISTICS")
        print("="*60)
        
        persons = {}
        
        for person_dir in self.output_dir.glob("person_*"):
            person_id = person_dir.name.split('_')[1]
            images = list(person_dir.glob("*.jpg"))
            
            ages = []
            for img in images:
                age = int(img.stem.split('_')[1])
                ages.append(age)
            
            ages = sorted(ages)
            
            persons[person_id] = {
                'count': len(images),
                'ages': ages,
                'age_range': (min(ages), max(ages)),
                'age_span': max(ages) - min(ages)
            }
        
        # Check if no persons found
        if len(persons) == 0:
            print("\nNo persons found in organized directory!")
            print(f"Make sure images are in: {self.source_dir}")
            return
        
        # Overall stats
        total_images = sum(p['count'] for p in persons.values())
        total_persons = len(persons)
        avg_images_per_person = total_images / total_persons
        
        print(f"\nTotal persons: {total_persons}")
        print(f"Total images: {total_images}")
        print(f"Avg images per person: {avg_images_per_person:.1f}")
        
        # Age span distribution
        age_spans = [p['age_span'] for p in persons.values()]
        print(f"\nAge span range: {min(age_spans)} - {max(age_spans)} years")
        print(f"Average age span: {sum(age_spans)/len(age_spans):.1f} years")
        
        # Show some examples
        print("\nSample persons:")
        for person_id in sorted(persons.keys())[:5]:
            info = persons[person_id]
            print(f"  Person {person_id}: {info['count']} images, "
                  f"ages {info['ages']}, span {info['age_span']} years")
        
        print("="*60)

# Usage
if __name__ == "__main__":
    # Organize dataset
    organizer = FGNETOrganizer(
        source_dir="./FGNET/FGNET/images",  # Raw FG-NET folder
        output_dir="./datasets/FGNET_organized"
    )
    
    # Step 1: Organize by person
    organizer.organize_by_person()
    
    # Step 2: Create test pairs
    organizer.create_test_pairs(min_age_gap=10)
    
    # Step 3: Print statistics
    organizer.print_statistics()
