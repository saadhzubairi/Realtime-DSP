from pathlib import Path
from typing import List


def read_text_with_fallback(file_path: Path) -> str:
    """Read text using UTF-8, falling back to latin-1 if needed.

    Some demo files may contain characters not valid in UTF-8 on Windows.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


def find_python_files(root_dir: Path) -> List[Path]:
    """Return a sorted list of .py files under root_dir (recursively),
    excluding common transient directories like __pycache__.
    """
    all_py_files = [
        p for p in root_dir.rglob("*.py") if "__pycache__" not in p.parts
    ]
    # Sort by normalized POSIX-like path for stable, human-friendly ordering
    return sorted(all_py_files, key=lambda p: p.as_posix().lower())


def write_combined_text(root_dir: Path, py_files: List[Path], output_file: Path) -> None:
    """Write all files into a single text file with headers:

    ===> <relative_path>
    <file content>
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out_f:
        for file_path in py_files:
            rel_path = file_path.relative_to(root_dir).as_posix()
            out_f.write(f"===> {rel_path}\n")
            content = read_text_with_fallback(file_path)
            out_f.write(content)
            if not content.endswith("\n"):
                out_f.write("\n")
            out_f.write("\n")  # blank line between files


def main() -> None:
    # Use the directory that contains this script as the root to search
    root_dir = Path(__file__).parent.resolve()
    output_file = root_dir / "all_python_files_combined.txt"

    py_files = find_python_files(root_dir)
    write_combined_text(root_dir, py_files, output_file)
    print(f"Wrote {len(py_files)} files to {output_file}")


if __name__ == "__main__":
    main()


