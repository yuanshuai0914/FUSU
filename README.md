FUSU: A Multi-temporal-source Land Use Change Segmentation Dataset for Fine-grained Urban Semantic Understanding

Baidu Ndisk:
[sample download](https://pan.baidu.com/s/1u7A_duHj61O507xnUWZSDg?pwd=s7um)
code: s7um 

Google Drive:
[sample download](https://drive.google.com/file/d/1x6pJO9IT4STzTlG85UKCvx3qudBgan9Q/view?usp=drive_link)

[Full data download](https://data-starcloud.pcl.ac.cn/zh/resource/57)

![image](https://github.com/yuanshuai0914/FUSU/blob/main/image/fusu.png)

This is the dataset and code for FUSU. (Coming soon. Please wait.)

About:

FUSU dataset covers 5 whole urban areas, 847 km^2 located in the north and south of China, with 17 land use and land cover (LULC) classes and over 170K images and 30 billion pixels of annotations, supporting segmentation, change detection and domain adaptation tasks. This data comprises 2 parts: 

1. Bi-temporal high-resolution satellite RGB images with fine-grained annotations.
  
2. Monthly revisited Sentinel-2 and Sentinel-1 images.


Details:

1. Resolution

The spatial resolution of high-resolution image is 0.2-0.5m, and the Sentinel image is 10m.
The time resolution of high-resolution image is 2 years, and the Sentinel image is one month.

2. Construction

FUSU comprises 62,752 image patches, each containing 25 images collected from Sentinel, 2 images from Google Earth, and 2 corresponding annotations. 

![image](https://github.com/yuanshuai0914/FUSU/blob/main/image/overall.png)

Example:
T1:       im1/6_255.png, im1_label/6_255.png
T2:       im2/6_255.png, im2_label/6_255.png
Sentinel: A/{8-12}/6_255_A_{8-12}.png, B/{1-12}/6_255_B_{1-12}.png, C/{1-12}/6_255_C_{1-12}.png

{A,B,C} represents the year, and {1-12} represents the month.

Shape: T1 image (512 times 512 times 3), T2 image (512 times 512 times 3), T1 label (512 times 512 times 1), T2 label (512 times 512 times 1),
Sentinel image (128 times 128 times 14).

14 bands of Sentinel images include 12 bands for Sentinel-2 and 2 bands (VV/VH) for Sentinel-1. Now they are concatenated, we will separate them into single .npy file in the future for easier usage.

3. Annotation

FUSU has 17 land use classes according to the Chinese Land Use Classification Criteria (GB/T21010-2017) Level-1 classification system.

FUSU_class = {
'0':'background','1':'traffic land','2':'inland water','3':'residential land','4':'cropland','5':'agriculture construction','6':'blank',
'7':'industrial land','8':'orchard','9':'park','10':'public management and service','11':'commercial land','12':'public construction',
'13':'special','14':'forest','15':'storage','16':'wetland','17':'grass'
}

PALETTE = {[255, 255, 255],[233, 133, 133],[8, 514, 230],[255, 0, 30],[126, 211, 33],[135, 126, 20],[94, 47, 4],[10, 82, 77],
[184, 233, 134],[219, 170, 230],[255, 199, 2],[252, 232, 5],[245, 107, 0],[243, 229, 176],[3, 100, 0],[127, 123, 127],[52, 205, 249],[18, 227, 180]
}

4. Citation

Yuan, S., Lin, G., Zhang, L., Dong, R., Zhang, J., Chen, S., ... & Fu, H. (2024). FUSU: A Multi-temporal-source Land Use Change Segmentation Dataset for Fine-grained Urban Semantic Understanding. arXiv preprint arXiv:2405.19055.
