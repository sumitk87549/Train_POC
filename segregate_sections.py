#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

def segregate_sections(input_file, output_dir):
    """
    Segregate text file into sections based on ===SECTION TITLE=== format
    and save each section as a separate file.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    sections = []
    current_title = None
    current_content = []
    
    for line in lines:
        line = line.rstrip('\n')
        
        # Check if line is a section header
        if line.startswith('===') and line.endswith('==='):
            # Save previous section if exists
            if current_title and current_content:
                sections.append((current_title, '\n'.join(current_content)))
            
            # Start new section
            current_title = line.strip('=== ').strip()
            current_content = []
        else:
            # Add to current content
            if current_title:  # Only add content if we have a section
                current_content.append(line)
    
    # Save last section
    if current_title and current_content:
        sections.append((current_title, '\n'.join(current_content)))
    
    section_count = 0
    
    for title, content in sections:
        if not title.strip() or not content.strip():
            continue
        
        # Clean filename - remove invalid characters
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_title:
            safe_title = f"section_{section_count + 1}"
        
        # Create filename
        filename = f"{safe_title.replace(' ', '_')}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Write section to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== {title} ===\n\n")
                f.write(content)
            
            print(f"Created: {filename}")
            section_count += 1
            
        except Exception as e:
            print(f"Error writing {filename}: {e}")
    
    print(f"\nSegregated {section_count} sections to '{output_dir}'")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Segregate text file into sections based on ===SECTION TITLE=== format"
    )
    parser.add_argument(
        'input_file',
        help='Input text file to segregate'
    )
    parser.add_argument(
        '-o', '--output',
        default='PROCESSED',
        help='Output directory (default: PROCESSED)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    success = segregate_sections(args.input_file, args.output)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
