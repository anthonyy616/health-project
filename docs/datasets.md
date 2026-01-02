# Dataset Download Guide

This guide provides instructions for downloading all datasets required for training the vital signs monitoring system.

> [!IMPORTANT]
> All datasets should be downloaded to the `data/raw/` directory. The total size is approximately **15-20 GB**.

---

## 📊 Required Datasets

| Dataset | Purpose | Size | Format |
|---------|---------|------|--------|
| UTKFace | Age estimation | ~2 GB | Images (JPG) |
| UBFC-rPPG | Heart rate (rPPG) | ~3 GB | Videos (AVI) + Ground truth |
| VIPL-HR | Heart rate (rPPG) | ~5 GB | Videos + labels |
| LFW (optional) | Face recognition | ~200 MB | Images |

---

## 1. UTKFace Dataset (Age Estimation)

**Purpose**: Training the age estimation model  
**Size**: ~2 GB  
**Contents**: 20,000+ face images labeled with age, gender, ethnicity

### Download Instructions

1. **Official Website**: https://susanqq.github.io/UTKFace/

2. **Direct Download (Kaggle)**:
   - Go to: https://www.kaggle.com/datasets/jangedoo/utkface-new
   - Click "Download" (requires Kaggle account)
   - Or use Kaggle CLI:
   ```powershell
   # Install Kaggle CLI (if not installed)
   pip install kaggle
   
   # Configure API (put kaggle.json in ~/.kaggle/)
   # Download from: https://www.kaggle.com/settings → Create New Token
   
   # Download dataset
   kaggle datasets download -d jangedoo/utkface-new -p data/raw/utkface --unzip
   ```

3. **File Naming Format**: `age_gender_race_date.jpg`
   - Example: `25_0_2_20170116174525125.jpg` = 25 years old, male, white

4. **Expected Structure**:
   ```
   data/raw/utkface/
   ├── 1_0_0_20161219140623097.jpg
   ├── 1_0_0_20161219140627985.jpg
   └── ... (20,000+ images)
   ```

---

## 2. UBFC-rPPG Dataset (Heart Rate)

**Purpose**: Training remote photoplethysmography (rPPG) for heart rate detection  
**Size**: ~3 GB  
**Contents**: 42 subjects, videos with ground truth PPG signals

### Download Instructions

1. **Official Website**: https://sites.google.com/view/yaboromance/ubfc-rppg

2. **Request Access**:
   - Fill out the form on the website
   - You'll receive a download link via email (usually within 24-48 hours)
   - Academic email addresses may get faster approval

3. **Alternative**: PURE Dataset (similar, easier access)
   - Website: https://www.tu-ilmenau.de/neurob/data-sets-code/pure-dataset/
   - Smaller (10 subjects) but good for initial experiments

4. **Expected Structure**:
   ```
   data/raw/ubfc-rppg/
   ├── subject1/
   │   ├── vid.avi          # Face video
   │   └── ground_truth.txt  # PPG signal + BPM
   ├── subject2/
   └── ... (42 subjects)
   ```

---

## 3. VIPL-HR Dataset (Heart Rate - Advanced)

**Purpose**: More diverse heart rate training data  
**Size**: ~5 GB  
**Contents**: 107 subjects, multiple scenarios (lighting, motion)

### Download Instructions

1. **Official Website**: https://vipl.ict.ac.cn/en/resources/databases/201901/t20190104_32243.html

2. **Request Access**:
   - Send email to the dataset maintainers (see website)
   - Include: name, affiliation, research purpose
   - Sign the license agreement they send back

3. **Expected Structure**:
   ```
   data/raw/vipl-hr/
   ├── p1/
   │   ├── v1/
   │   │   ├── source1/  # Different cameras
   │   │   └── source2/
   │   └── v2/
   └── ... (107 subjects)
   ```

---

## 4. FLIR Thermal Dataset (Temperature - Optional)

**Purpose**: Thermal imaging for temperature estimation  
**Size**: ~1 GB  

### Download Instructions

1. **FLIR ADAS Dataset**: https://www.flir.com/oem/adas/adas-dataset-form/
   - Fill out the form (free for research)
   
2. **Alternative**: Use synthetic thermal + calibration data
   - We'll generate calibration data using the MLX90614 sensor

---

## 🛠️ Download Script

Run this script to check your data directory:

```python
# scripts/check_datasets.py
import os
from pathlib import Path

def check_datasets():
    data_dir = Path("data/raw")
    
    datasets = {
        "utkface": {"min_files": 1000, "desc": "Age estimation images"},
        "ubfc-rppg": {"min_files": 10, "desc": "Heart rate videos"},
        "vipl-hr": {"min_files": 10, "desc": "Advanced HR videos"},
    }
    
    print("Dataset Status:")
    print("=" * 50)
    
    for name, info in datasets.items():
        path = data_dir / name
        if path.exists():
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            status = "✅" if count >= info["min_files"] else "⚠️"
            print(f"{status} {name}: {count} files - {info['desc']}")
        else:
            print(f"❌ {name}: Not found - {info['desc']}")
    
    print("=" * 50)

if __name__ == "__main__":
    check_datasets()
```

---

## 📝 Notes

1. **Storage**: Ensure you have at least **25 GB** free space

2. **Processing**: After download, run preprocessing scripts to:
   - Resize images to consistent dimensions
   - Extract frames from videos
   - Normalize labels

3. **Backup**: Consider uploading processed data to your Supabase for backup

4. **Citation**: Remember to cite datasets in your thesis:
   ```
   @article{utkface,
     title={Age Estimation and Gender Classification},
     author={Zhang, Zhifei and Song, Yang and Qi, Hairong}
   }
   
   @inproceedings{ubfc-rppg,
     title={UBFC-RPPG: A Public Benchmark Dataset for Remote Heart Rate Estimation},
     author={Bobbia, S. and Macwan, R. and Benezeth, Y.}
   }
   ```

---

## ❓ FAQ

**Q: Can I start training without all datasets?**  
A: Yes! Start with UTKFace for age estimation - it's the easiest to obtain.

**Q: What if dataset access is denied?**  
A: Use alternatives like Kaggle versions or smaller public datasets.

**Q: How long do approvals take?**  
A: Academic datasets: 1-7 days. Commercial: instant (needs registration).
