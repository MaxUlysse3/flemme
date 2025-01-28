use rand::Rng;

use ndarray::{
    Array1,
    Array2,
    Axis,
};

pub struct Layer(pub usize);

pub struct NetworkBuilder {
    pub layers: Vec<Layer>,
}

impl NetworkBuilder {
    pub fn build(self) -> Result<Network, String> {
        if self.layers.len() == 0 {
            return Err("No layers.".to_string());
        }
        let mut res = Network {
            weights: vec![],
            biases: vec![],
        };

        let mut last_size = self.layers[0].0;
        
        for (i, Layer(current_size)) in self.layers.into_iter().enumerate() {
            if i == 0 { continue; }

            let w = Array2::<f64>::from_shape_fn((current_size, last_size), |(_, _)| {
                rand::thread_rng().gen_range(0..1000) as f64 / 1000f64
            });
            let b = Array1::<f64>::from_shape_fn(current_size, |_| rand::thread_rng().gen_range(0..1000) as f64 / 1000f64);
            res.weights.push(w);
            res.biases.push(b);
            last_size = current_size;
        }
        Ok(res)
    }
}

pub struct Network {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

fn sigma(x: f64) -> f64 {
    1f64 / (1f64 + f64::exp(-x))
}

fn sigma_vec(x: Array1<f64>) -> Array1<f64> {
    x.into_iter().map(sigma).collect::<Array1<f64>>()
}

fn sigma_prime_vec(x: Array1<f64>) -> Array1<f64> {
    x.into_iter().map(|x| sigma(x) * (1f64 - sigma(x))).collect::<Array1<f64>>()
}

impl Network {
    pub fn compute(&self, input: Array1<f64>) -> Array1<f64> {
        let mut res = input;
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            // println!("Res : {res:?} \nWeight : {w:?}\n");
            res = w.dot(&res);
            res += b;
            res = sigma_vec(res);
        }
    
        res
    }

    pub fn forward(&self, input: Array1<f64>) -> Vec<Array1<f64>> {
        let mut res = vec![];
        res.push(input.clone());
        let mut last = input;

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            // println!("Res : {res:?} \nWeight : {w:?}\n");
            let mut new_in = w.dot(&last);
            new_in += b;
            new_in = sigma_vec(new_in);
            res.push(new_in.clone());
            last = new_in;
        }

        res
    }

    pub fn backward(&self, input: Array1<f64>, target: Array1<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        let mut state = self.forward(input);
        let mut grad_w = vec![];
        let mut grad_b = vec![];
        let len = state.len();

        let mut delta = (state.last().unwrap().clone() - &target) * 2f64;
        // println!("delta_len : {:?}", delta.len());
        let mut der = sigma_prime_vec(state.last().unwrap().clone());
        // println!("der_len : {:?}", der.len());
        delta.iter_mut().enumerate().for_each(|(i, x)| *x = *x * der[i]);

        for (i, act) in state.iter().enumerate().rev().take(len - 1) {
            // println!("Computing delta : {i:?}");
            let last_act = &state[i - 1];
            grad_w.push(delta.clone().insert_axis(Axis(1)).dot(&last_act.view().insert_axis(Axis(0))));
            grad_b.push(delta.clone());

            delta = self.weights[i - 1].t().dot(&delta);
            der = sigma_prime_vec(state[i - 1].clone());
            delta.iter_mut().enumerate().for_each(|(i, x)| *x = *x * der[i])
        }
        grad_w.reverse();
        grad_b.reverse();

        (grad_w, grad_b)
    }

    pub fn gradient(&self, mut data: Vec<(Array1<f64>, Array1<f64>)>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        let n = data.len() as f64;

        let first_dat = data.pop().expect("Dataset is empty.");
        let (mut grad_w, mut grad_b) = self.backward(first_dat.0, first_dat.1);

        for (i, t) in data.into_iter() {
            let (gw, gb) = self.backward(i, t);
            grad_w.iter_mut().zip(gw.iter()).for_each(|(g, w)| *g += w);
            grad_b.iter_mut().zip(gb.iter()).for_each(|(g, b)| *g += b);
        }

        grad_w.iter_mut().for_each(|x| *x /= n);
        grad_b.iter_mut().for_each(|x| *x /= n);

        (grad_w, grad_b)
    }

    pub fn cost(&self, input: Array1<f64>, target: Array1<f64>) -> f64 {
        let out = self.compute(input);
        out.into_iter().zip(target.into_iter()).map(|(x, t)| (x - t) * (x - t)).sum()
    }

    pub fn learn(&mut self, dataset: impl Iterator<Item = (Array1<f64>, Array1<f64>)> + Clone,
        sample_size: usize, num_training: usize) {
        let mut iter = dataset.cycle();
        let mut fac_grad = 6.;
        for i in 0..num_training {
            let mut data = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                data.push(iter.next().unwrap());
            }
            if i % (10000 / sample_size) == 0 {
                println!("Learning loop : {i:?} / fac_grad : {fac_grad}");
                let (input, target) = data.last().unwrap().clone();
                let c = self.cost(input, target);
                println!("Cost : {:?}", c);
                // if c <= 0.2 {
                //     fac_grad = if fac_grad > 2. { 2. } else { fac_grad };
                // }
            }
            let (mut grad_w, mut grad_b) = self.gradient(data);
            grad_w.iter_mut().for_each(|x| *x *= fac_grad);
            grad_b.iter_mut().for_each(|x| *x *= fac_grad);
            // println!("Updating gradient BEGINING");
            self.weights.iter_mut().zip(grad_w.iter()).for_each(|(w, g)| *w -= g);
            self.biases.iter_mut().zip(grad_b.iter()).for_each(|(b, g)| *b -= g);
            // println!("Updating gradient END\n\n\n");
            fac_grad /= 1.001;
        }
    }
}
