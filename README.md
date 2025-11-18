# scSID Reproducibility Package

To ensure the reproducibility of our experimental results during peer review stage, this repository provides the **complete Stage II annotation code** used in our paper. Our **scSID** framework includes optimizations for both **CIForm** and **scSFUT**, and we additionally provide:

- Five-fold splits of the **mat** dataset  
- Pretrained **Cell-SIDs** corresponding to each fold  
- Implementations capable of reproducing the following models:
  - **CIForm baseline**
  - **scSFUT baseline**
  - **CIForm + Cell-SID**
  - **scSFUT + Cell-SID**

Other components of the pipeline—including **Stage I code** and **additional extreme comparison models**—will be released after paper acceptance or upon request from reviewers.

---

## Contents

This repository currently includes:

- `results/preprocessed_samples/mat/` — Five-fold dataset splits  
- `ciform.py` — CIForm baseline and CIForm + Cell-SID implementations  
- `scsfut` and `scsfut+sid` — scSFUT baseline and scSFUT + Cell-SID implementations  
- `results/sid_indices/mat/` — Pretrained Cell-SIDs for each data fold  
- `my_model/` — Specific model pipeline of scSFUT.py and scSFUT+sid.py
- `CIForm_mat_base.log` and `CIForm_sid_mat.log` — Training logs reported in scSID paper
- `README.md`

---

## Acknowledgements

We sincerely thank the **CIForm** and **scSFUT** teams for their valuable open-source contributions to the single-cell annotation community.

### CIForm

- **Official Code:**  
  https://github.com/zhanglab-wbgcas/CIForm.git  

- **Citation:**  
  Xu J, Zhang A, Liu F, et al.  
  *CIForm as a transformer-based model for cell-type annotation of large-scale single-cell RNA-seq data.*  
  **Briefings in Bioinformatics**, 2023, 24(4): bbad195.

---

### scSFUT

- **Official Code:**  
  https://github.com/MagnificentZZZ/scSFUT.git  

- **Citation:**  
  Zhang H, Jiang Z, Zhang S, et al.  
  *Scale-free and unbiased transformer with tokenization for cell type annotation from single-cell RNA-seq data.*  
  **Pattern Recognition**, 2025: 111724.

---

## Notes

If you have any questions or require additional components, please contact us through the Issue Tracker or Email.
