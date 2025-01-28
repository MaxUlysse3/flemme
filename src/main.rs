#![feature(iter_next_chunk)]

mod network;
mod reader;

use ndarray::{
    Array1,
    Array2,
    arr1,
};

use reader:: {
    Image,
    Set,
    ImageIter,
};

use rand::Rng;

use network::{
    Network,
    NetworkBuilder,
    Layer,
};

fn main() { 
    let s = vec![Layer(784), Layer(16), Layer(16), Layer(10)];
    let mut n = NetworkBuilder { layers: s }.build().unwrap();

    let images = ImageIter::new(Set::Train);

    let dataset = images.map(|i| i.to_data()).collect::<Vec<_>>();
    n.learn(dataset.into_iter(), 100, 10000);

    let mut accurate = 0;
    let test_num = 10000;
    for (i, t) in ImageIter::new(Set::Test).map(|x| x.to_data()).take(test_num) {
        let res = t.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        let out = n.compute(i).iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        if res == out {
            accurate += 1;
        }
    }
    println!("Accuracy : {:?}", accurate as f64 / test_num as f64);

}

fn gen_data(num: usize) -> Vec<(Array1<f64>, Array1<f64>)> {
    let f = |x: f64| x;
    let iter = std::iter::from_fn(|| {
        let x = rand::thread_rng().gen_range(0..1000) as f64 / 1000f64;
        let y = rand::thread_rng().gen_range(0..1000) as f64 / 1000f64; 
        let out = if y > f(x) { 1.0 } else { 0.0 };
        Some((arr1(&[x, y]), arr1(&[out])))
    });
    iter.take(num).collect::<Vec<_>>()
}
