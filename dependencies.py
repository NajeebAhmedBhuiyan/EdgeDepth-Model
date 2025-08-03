import importlib
import subprocess
import sys
from packaging import version

def verify_package(package_name, min_version=None):
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', None)
        
        if installed_version:
            status = f"✓ {package_name} ({installed_version})"
            if min_version and version.parse(installed_version) < version.parse(min_version):
                status += f" - WARNING: Below recommended version {min_version}"
            print(status)
            return True
        else:
            print(f"✓ {package_name} (version unknown)")
            return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

# Critical packages with minimum recommended versions
critical_packages = {
    "numpy": "1.18.0",
    "cv2": "4.5.0",  # opencv-python
    "matplotlib": "3.3.0",
    "tensorflow": "2.10.0",
    "ultralytics": "8.0.0",
    "typeguard": "2.13.0"  # For conflict resolution
}

# Additional important packages
other_packages = [
    "PIL",  # Pillow
    "requests",
    "protobuf",
    "torch",
    "torchvision",
    "pandas",
    "scipy"
]

print("="*50)
print("Verifying Critical Dependencies")
print("="*50)
all_critical_ok = True
for package, min_ver in critical_packages.items():
    if not verify_package(package, min_ver):
        all_critical_ok = False

print("\n" + "="*50)
print("Verifying Other Important Dependencies")
print("="*50)
for package in other_packages:
    verify_package(package)

print("\n" + "="*50)
print("Checking Dependency Conflict Resolution")
print("="*50)
try:
    import generate_parameter_library_py
    try:
        import typeguard
        print("✓ typeguard installed - conflict resolved")
    except ImportError:
        print("✗ typeguard MISSING - conflict not resolved!")
        print("  Run: pip install typeguard")
except ImportError:
    print("⚠ generate_parameter_library_py not found - conflict irrelevant")

print("\n" + "="*50)
print("Environment Summary")
print("="*50)
print(f"Python: {sys.version}")
print(f"Virtual Environment: {'object-detection-env' in sys.executable}")

# GPU Availability Check (for TensorFlow and PyTorch)
print("\n" + "="*50)
print("Hardware Acceleration Check")
print("="*50)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPU: {'Available' if gpus else 'Not Available'}")
except ImportError:
    print("TensorFlow not available for GPU check")

try:
    import torch
    print(f"PyTorch GPU: {'Available' if torch.cuda.is_available() else 'Not Available'}")
except ImportError:
    print("PyTorch not available for GPU check")

print("\n" + "="*50)
if all_critical_ok:
    print("✅ ALL CRITICAL DEPENDENCIES VERIFIED SUCCESSFULLY")
else:
    print("❌ SOME DEPENDENCIES MISSING OR OUTDATED")