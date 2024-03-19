<h1 align="center">
ChemSAM
</h1>

<h3 align="center">
Automated Molecular Structure Segmentation from Documents Using ChemSAM
</h3> 

<img src="image/demo_image.png" align="center">

---

# Abstract
Chemical structure segmentation constitutes a pivotal task in cheminformatics, involving the extraction and abstraction of structural information of chemical compounds from text-based sources, including patents and scientific articles. This study introduces a deep learning approach to chemical structure segmentation, employing a Vision Transformer (ViT) to discern the structural patterns of chemical compounds from their graphical representations. The Chemistry-Segment Anything Model (ChemSAM) achieves state-of-the-art results on publicly available benchmark datasets and real-world tasks, underscoring its effectiveness in accurately segmenting chemical structures from text-based sources. Moreover, this deep learning-based approach obviates the need for handcrafted features and demonstrates robustness against variations in image quality and style. During the detection phase, a ViT-based encoder-decoder model is used to identify and locate chemical structure depictions on the input page. This model generates masks to ascertain whether each pixel belongs to a chemical structure, thereby offering a pixel-level classification and indicating the presence or absence of chemical structures at each position. Subsequently, the generated masks are clustered based on their connectivity, and each mask cluster is updated to encapsulate a single structure in the post-processing workflow. This two-step process facilitates the effective automatic extraction of chemical structure depictions from documents. By utilizing the deep learning approach described herein, it is demonstrated that effective performance on low-resolution and densely arranged molecular structural layouts in journal articles and patents is achievable. The free download paper link is available at: [Releases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10935819/)

# User Manual

- [PharmaMind-MolMiner User Guide](./docs/en-US/PharmaMind%20User%20Guide.pdf) in English.

- [PharmaMind-MolMiner User Guide](./docs/zh-CN/PharmaMind%20User%20Guide.pdf) in Chinese.

# Release logs

- [PharmaMind-MolMiner Releases](https://github.com/iipharma/pharmamind-molminer/releases)
- [MAC CDN download](https://molminer-cdn.iipharma.cn/pharma-mind/artifact/latest/mac/PharmaMind-mac-latest-setup.dmg)
- [Win CDN download](https://molminer-cdn.iipharma.cn/pharma-mind/artifact/latest/win/PharmaMind-win-latest-setup.exe)
