use ai_dataloader::{Dataset, GetSample, Len};
use image::{self, GenericImageView};
use ndarray::{azip, s, Array3};
use nshare::ToNdarray3;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

#[derive(Serialize, Deserialize, Debug)]
struct MetaData {
    classes: Vec<Vec<String>>,
    class_to_idx: HashMap<String, i64>,
    samples: Vec<(String, i64)>,
}

#[derive(Debug, Clone)]
pub struct Transforms {
    mean: (f32, f32, f32),
    std: (f32, f32, f32),
    crop_size: u32,
    resize_size: u32,
    interpolation: image::imageops::FilterType,
}

impl Transforms {
    pub fn new(
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
        crop_size: u32,
        resize_size: u32,
        interpolation: image::imageops::FilterType,
    ) -> Self {
        Transforms {
            mean,
            std,
            crop_size,
            resize_size,
            interpolation,
        }
    }
}

#[derive(Debug)]
pub struct ImageNetDataset {
    root_dir: PathBuf,
    metadata: MetaData,
    transforms: Transforms,
}

impl ImageNetDataset {
    pub fn new(root_dir: PathBuf, transforms: Transforms) -> Self {
        let mut file = File::open("examples/vision/imagenet_val.json").expect("imagenet_val.json file not found");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Something went wrong reading the file, imagenet_val.json");
        let metadata = serde_json::from_str(&content).expect("imagenet_val.json, was not well-formatted");
        ImageNetDataset {
            root_dir,
            metadata,
            transforms,
        }
    }
}

impl Dataset for ImageNetDataset {}
impl Len for ImageNetDataset {
    fn len(&self) -> usize {
        self.metadata.samples.len()
    }
}

impl GetSample for ImageNetDataset {
    type Sample = (Array3<f32>, i64);
    fn get_sample(&self, idx: usize) -> Self::Sample {
        let resize_size = self.transforms.resize_size;
        let crop_size = self.transforms.crop_size;
        let mean = self.transforms.mean;
        let mean = [mean.0, mean.1, mean.2];
        let std = self.transforms.std;
        let vec = [std.0, std.1, std.2];
        let std = vec;

        let (path, label) = &self.metadata.samples[idx];

        let img = image::open(self.root_dir.join(path))
            .unwrap_or_else(|_| 
                panic!("{:?} not found, make sure you have the correct path and imagenet dataset downloaded", self.root_dir.join(path)))
            .resize_to_fill(
                resize_size,
                resize_size,
                self.transforms.interpolation,
            );
        // CenterCrop
        let (width, height) = img.dimensions();
        let start_x = if width > crop_size {
            (width - crop_size) / 2
        } else {
            0
        };
        let start_y = if height > crop_size {
            (height - crop_size) / 2
        } else {
            0
        };
        let img = img.crop_imm(start_x, start_y, crop_size, crop_size);

        let img = img.into_rgb32f().into_ndarray3();
        // Normalize each channel separately
        let mut normalized_img = img.clone();
        for channel in 0..3 {
            let slice = img.slice(s![channel, .., ..]);
            let norm_slice = normalized_img.slice_mut(s![channel, .., ..]);
            let mean = mean[channel];
            let std_dev = std[channel];
            azip!((norm in norm_slice, &val in slice) *norm = (val - mean) / std_dev);
        }

        (normalized_img, *label)
    }
}
