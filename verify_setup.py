#!/usr/bin/env python3
"""
Setup Verification Script
Author: Deepak Kumar
Course: INFO7375 - Fall 2025

This script verifies that all dependencies are installed correctly.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """
    Check if a module can be imported.
    
    Args:
        module_name: Name of module to import
        package_name: Display name (if different from module_name)
    
    Returns:
        bool: True if import successful
    """
    if package_name is None:
        package_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package_name:20s} (version {version})")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - NOT INSTALLED")
        print(f"   Error: {e}")
        return False


def main():
    """Run all checks."""
    print("="*70)
    print("SETUP VERIFICATION")
    print("="*70)
    print()
    
    print("Checking Python version...")
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print("❌ Python 3.8 or higher required!")
        return False
    else:
        print("✅ Python version OK")
    
    print()
    print("Checking dependencies...")
    print("-"*70)
    
    # Core dependencies
    results = []
    results.append(check_import('torch', 'PyTorch'))
    results.append(check_import('numpy', 'NumPy'))
    results.append(check_import('gymnasium', 'Gymnasium'))
    results.append(check_import('ale_py', 'ALE-Py'))
    
    # Visualization
    results.append(check_import('matplotlib', 'Matplotlib'))
    results.append(check_import('seaborn', 'Seaborn'))
    results.append(check_import('cv2', 'OpenCV'))
    
    # Utilities
    results.append(check_import('pandas', 'Pandas'))
    results.append(check_import('yaml', 'PyYAML'))
    results.append(check_import('tqdm', 'tqdm'))
    
    # Optional
    results.append(check_import('tensorboard', 'TensorBoard'))
    results.append(check_import('PIL', 'Pillow'))
    results.append(check_import('scipy', 'SciPy'))
    
    print("-"*70)
    
    # Check device availability
    print()
    print("Checking compute devices...")
    print("-"*70)
    
    import torch
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        cuda_available = True
    else:
        print("⚠️  CUDA not available (CPU only)")
        cuda_available = False
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS available (Apple Silicon GPU)")
        mps_available = True
    else:
        print("⚠️  MPS not available")
        mps_available = False
    
    if not cuda_available and not mps_available:
        print("ℹ️  Will use CPU (training will be slower)")
    
    print("-"*70)
    
    # Test Gymnasium environment
    print()
    print("Testing Gymnasium Atari environment...")
    print("-"*70)
    
    try:
        import gymnasium as gym
        env = gym.make('ALE/Jamesbond-v5')
        obs, info = env.reset()
        print(f"✅ Environment created successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"❌ Failed to create environment")
        print(f"   Error: {e}")
        print()
        print("   Try installing Atari ROMs:")
        print("   pip install 'gymnasium[atari,accept-rom-license]'")
        results.append(False)
    
    print("-"*70)
    
    # Test local imports
    print()
    print("Testing local modules...")
    print("-"*70)
    
    sys.path.append('src')
    
    local_modules = [
        'network',
        'dqn_agent',
        'replay_buffer',
        'preprocessing',
        'trainer',
        'utils'
    ]
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module:20s}")
            results.append(True)
        except Exception as e:
            print(f"❌ {module:20s} - Error: {e}")
            results.append(False)
    
    print("-"*70)
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    if all(results):
        print()
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("You're ready to start training:")
        print("  python src/run_experiments.py --experiment quick_test")
        print()
        return True
    else:
        print()
        print("⚠️  SOME CHECKS FAILED")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("  pip install 'gymnasium[atari,accept-rom-license]'")
        print()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
