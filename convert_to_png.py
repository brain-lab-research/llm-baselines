#!/usr/bin/env python3
"""
Convert PDF figures to PNG for better GitHub/README display
"""

import subprocess
import sys
from pathlib import Path

def install_package(package_name):
    """Install a Python package"""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

def convert_pdfs_to_png():
    """Convert all PDFs in figures directories to PNG"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not found. Installing...")
        install_package("PyMuPDF")
        import fitz
    
    from pathlib import Path
    
    base_dir = Path(__file__).parent
    dirs_to_process = [
        base_dir / "figures_smoothness",
        base_dir / "figures_main"
    ]
    
    converted_count = 0
    
    for dir_path in dirs_to_process:
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            continue
        
        print(f"\n📁 Processing {dir_path.name}/")
        pdf_files = sorted(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"   No PDF files found")
            continue
        
        for pdf_file in pdf_files:
            try:
                png_file = pdf_file.with_suffix(".png")
                
                # Open PDF and convert first page
                doc = fitz.open(pdf_file)
                page = doc[0]
                
                # Render with high quality (300 DPI)
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Save PNG
                pix.save(png_file)
                doc.close()
                
                print(f"   ✅ {pdf_file.name} → {png_file.name}")
                converted_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed to convert {pdf_file.name}: {e}")
    
    if converted_count > 0:
        print(f"\n✨ Successfully converted {converted_count} PDFs to PNG!")
        print("Now you can update README_PAPER.md to use .png instead of .pdf")
        return True
    else:
        print("❌ No PDFs were converted")
        return False

if __name__ == "__main__":
    success = convert_pdfs_to_png()
    sys.exit(0 if success else 1)
