use std::{
    fs,
    iter,
};

use serde::de::value;

#[derive(Clone)]
pub struct Image {
    image: Vec<u8>,
    label: u8,
}

impl Image {
    fn new(image: Vec<u8>, label: u8) -> Self {
        Self {
            image,
            label,
        }
    }

    #[inline]
    pub fn image(self) -> Vec<u8> {
        self.image
    }

    #[inline]
    pub fn label(&self) -> u8 {
        self.label
    }

    pub fn show(&self) {
        let mut display = String::new();
        for i in 0..28 {
            for j in 0..28 {
                display.push(match self.image[28 * i + j] {
                    0..=80 => ' ',
                    81..=120 => '.',
                    121..=180 => '/',
                    181..=252 => '$',
                    253..=255 => '#',
                });
            }
            display.push('\n');
        }

        println!("{}", display);
    }

    pub fn to_data(self) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
        self.into()
    }
}

impl From<Image> for (Vec<u8>, u8) {
    #[inline]
    fn from(value: Image) -> Self {
        (value.image, value.label)
    }
}

impl From<Image> for Vec<u8> {
    #[inline]
    fn from(value: Image) -> Self {
        value.image()
    }
}

impl From<Image> for ndarray::Array1<f64> {
    #[inline]
    fn from(value: Image) -> Self {
        let vector: Vec<u8> = value.into();
        Self::from_shape_fn(vector.len(), |x| vector[x].into())
    }
}

impl From<Image> for (ndarray::Array1<f64>, ndarray::Array1<f64>) {
    #[inline]
    fn from(value: Image) -> Self {
        let mut target = vec![0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9];
        for i in target.iter_mut() {
            *i = if *i == value.label {1} else {0};
        }
        let vector: Vec<u8> = value.image;
        (ndarray::Array1::from_shape_fn(vector.len(), |x| vector[x].into()), ndarray::Array1::from_shape_fn(target.len(), |x| target[x].into()))
    }
}

pub enum Set {
    Train,
    Test,
}

// const PATH: &str = "../mnist/{}-{}.idx";

type VecIntoIter<T> = <Vec<T> as IntoIterator>::IntoIter;
pub struct ImageIter {
    data_img: VecIntoIter<u8>,
    data_lab: VecIntoIter<u8>,
}

impl ImageIter {
    pub fn new(set: Set) -> Self {
        let set_str = match set {
            Set::Train => "train",
            Set::Test => "test",
        };
        let mut to_return = Self {
            data_img: fs::read(format!("../mnist/{}-{}.idx", set_str, "images")).expect("File not found.").into_iter(),
            data_lab: fs::read(format!("../mnist/{}-{}.idx", set_str, "labels")).expect("File not found.").into_iter(),
        };
        
        println!("{:?}", to_return.data_img.next_chunk::<16>().unwrap());
        println!("{:?}", to_return.data_lab.next_chunk::<8>().unwrap());
        to_return
    }
}

impl Iterator for ImageIter {
    type Item = Image;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.data_img.next_chunk::<784>() {
            Err(_) => None,
            Ok(v) => match self.data_lab.next() {
                None => None,
                Some(l) => Some(Image::new(v.into(), l))
            }
        }
    }
}
