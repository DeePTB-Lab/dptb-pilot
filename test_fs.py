import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dptb_agent_tools.modules.fs_tool import list_directory, read_file_content

def test_fs_tools():
    print("Testing list_directory...")
    # List current directory
    cwd = os.getcwd()
    print(f"Listing {cwd}")
    ls_result = list_directory(cwd)
    print(ls_result[:200] + "..." if len(ls_result) > 200 else ls_result)
    
    if "Contents of" in ls_result:
        print("✅ list_directory passed")
    else:
        print("❌ list_directory failed")

    print("\nTesting read_file_content (Text)...")
    # Read README.md
    readme_path = os.path.join(cwd, "README.md")
    read_result = read_file_content(readme_path)
    print(f"Reading {readme_path}")
    print(read_result[:100] + "..." if len(read_result) > 100 else read_result)
    
    if "# dptb-pilot" in read_result or "DeePTB" in read_result:
        print("✅ read_file_content (Text) passed")
    else:
        print("❌ read_file_content (Text) failed")
        
    # Test PDF reading if a PDF exists in temp/pdfs
    pdf_dir = os.path.join(cwd, "temp", "pdfs")
    if os.path.exists(pdf_dir):
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if pdfs:
            print("\nTesting read_file_content (PDF)...")
            pdf_path = os.path.join(pdf_dir, pdfs[0])
            print(f"Reading {pdf_path}")
            pdf_result = read_file_content(pdf_path)
            print(pdf_result[:200] + "..." if len(pdf_result) > 200 else pdf_result)
            
            if "PDF Content" in pdf_result:
                print("✅ read_file_content (PDF) passed")
            else:
                print("❌ read_file_content (PDF) failed")
        else:
            print("\n⚠️ No PDFs found to test.")
    else:
        print("\n⚠️ PDF directory not found.")

if __name__ == "__main__":
    test_fs_tools()
