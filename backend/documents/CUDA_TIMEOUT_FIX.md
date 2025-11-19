# Fix for CUDA Watchdog Timeout Error

## Problem
```
CUDA error: the launch timed out and was terminated
```

## Root Cause
The NVIDIA X server has a watchdog timer that kills CUDA kernels running longer than ~5 seconds to prevent display freezing.

## Solution: Disable Watchdog Timer

### Option 1: Add to Xorg Configuration (Recommended)

Edit `/etc/X11/xorg.conf` and add `Option "Interactive" "0"` to each Device section:

```bash
sudo nano /etc/X11/xorg.conf
```

Add this line to **each** `Section "Device"`:

```
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BoardName      "NVIDIA GeForce RTX 3060"
    BusID          "PCI:193:0:0"
    Option         "Interactive" "0"    # <-- ADD THIS LINE
EndSection

Section "Device"
    Identifier     "Device1"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BoardName      "NVIDIA GeForce RTX 3060"
    BusID          "PCI:2:0:0"
    Option         "Interactive" "0"    # <-- ADD THIS LINE
EndSection

# ... repeat for Device2, Device3, etc.
```

### Option 2: Use nvidia-smi to set persistence mode

```bash
sudo nvidia-smi -pm 1
```

### After Making Changes

1. **Restart X server** (logout and login) OR reboot
2. **Restart backend**:
   ```bash
   cd backend
   python main.py
   ```

## Verification

After restart, the CUDA timeout errors should be gone.

## Alternative: Use Only Non-Display GPUs

If you don't want to modify system settings, use only GPUs without displays:

```bash
# In backend/.env
QWEN_TRANSFORMERS_GPU_DEVICES=0,1,2,3  # Use GPUs without display
```

Check which GPUs have displays:
```bash
nvidia-smi --query-gpu=index,display_active --format=csv
```

