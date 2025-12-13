"""
Compile protocol buffer definitions to Python code.

This script compiles the mrlm.proto file into Python code using grpc_tools.
Run this script after modifying the .proto file:

    python -m mrlm.protocols.compile_proto

Generated files:
    - mrlm_pb2.py: Protocol buffer message classes
    - mrlm_pb2_grpc.py: gRPC service stubs and servers
"""

import subprocess
import sys
from pathlib import Path


def compile_proto():
    """Compile protocol buffer definitions."""
    # Get the protocols directory
    protocols_dir = Path(__file__).parent
    proto_file = protocols_dir / "mrlm.proto"

    if not proto_file.exists():
        print(f"Error: {proto_file} not found!")
        sys.exit(1)

    # Output directory (same as proto file location)
    output_dir = protocols_dir

    # Compile command
    # We use python -m grpc_tools.protoc instead of protoc directly
    # to ensure we use the version from the virtual environment
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={protocols_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",  # Generate type stubs
        str(proto_file),
    ]

    print("Compiling protocol buffers...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Successfully compiled protocol buffers")

        # List generated files
        print("\nGenerated files:")
        for pattern in ["*_pb2.py", "*_pb2_grpc.py", "*_pb2.pyi"]:
            for file in protocols_dir.glob(pattern):
                print(f"  - {file.name}")

        # Fix imports in generated files
        fix_imports(output_dir)

    except subprocess.CalledProcessError as e:
        print(f"✗ Error compiling protocol buffers:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Error: grpc_tools not found!")
        print("Install it with: pip install grpcio-tools")
        sys.exit(1)


def fix_imports(output_dir: Path):
    """
    Fix imports in generated files to use relative imports.

    The generated code uses absolute imports which may not work correctly.
    We convert them to relative imports.
    """
    print("\nFixing imports in generated files...")

    # Fix mrlm_pb2_grpc.py
    grpc_file = output_dir / "mrlm_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()

        # Replace absolute import with relative import
        old_import = "import mrlm_pb2 as mrlm__pb2"
        new_import = "from . import mrlm_pb2 as mrlm__pb2"

        if old_import in content:
            content = content.replace(old_import, new_import)
            grpc_file.write_text(content)
            print(f"  ✓ Fixed imports in {grpc_file.name}")
        else:
            print(f"  - No import fixes needed in {grpc_file.name}")


if __name__ == "__main__":
    compile_proto()
