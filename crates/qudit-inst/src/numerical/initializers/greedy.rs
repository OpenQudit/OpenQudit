use qudit_core::RealScalar;
use super::super::InitialGuessGenerator;
use rand::Rng;
use rand::distributions::Uniform as RandUniform;

#[derive(Clone)]
pub struct GreedyFurthestPoint<R: RealScalar> {
    lower_bound: R,
    upper_bound: R,
    num_candidates: usize,
    selected_points: Vec<Vec<R>>,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: RealScalar> GreedyFurthestPoint<R> {
    pub fn new(lower_bound: R, upper_bound: R, num_candidates: usize) -> Self {
        if lower_bound > upper_bound {
            panic!("Lower bound cannot be larger than upper bound.");
        }
        Self {
            lower_bound,
            upper_bound,
            num_candidates,
            selected_points: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.selected_points.clear();
    }

    fn generate_candidates(&self, num_params: usize) -> Vec<Vec<R>> {
        let mut rng = rand::thread_rng();
        let distribution = RandUniform::new(
            self.lower_bound.to64(),
            self.upper_bound.to64(),
        );

        (0..self.num_candidates)
            .map(|_| {
                (0..num_params)
                    .map(|_| R::from64(rng.sample(distribution)))
                    .collect()
            })
            .collect()
    }

    fn euclidean_distance(&self, point1: &[R], point2: &[R]) -> R {
        point1
            .iter()
            .zip(point2.iter())
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .fold(R::zero(), |acc, x| acc + x)
            .sqrt()
    }

    fn min_distance_to_selected(&self, candidate: &[R]) -> R {
        if self.selected_points.is_empty() {
            return R::from64(std::f64::INFINITY);
        }

        self.selected_points
            .iter()
            .map(|selected| self.euclidean_distance(candidate, selected))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(R::zero())
    }

    pub fn generate_multiple(&mut self, num_params: usize, num_points: usize) -> Vec<Vec<R>> {
        self.reset();
        
        if num_points == 0 {
            return Vec::new();
        }

        let candidates = self.generate_candidates(num_params);
        let mut result = Vec::with_capacity(num_points);

        // Select first point randomly
        let mut rng = rand::thread_rng();
        let first_idx = rng.gen_range(0..candidates.len());
        result.push(candidates[first_idx].clone());
        self.selected_points.push(candidates[first_idx].clone());

        // Greedily select remaining points
        for _ in 1..num_points {
            let mut best_candidate_idx = 0;
            let mut best_min_distance = R::zero();

            for (idx, candidate) in candidates.iter().enumerate() {
                // Skip if already selected
                if self.selected_points.iter().any(|selected| {
                    selected.iter().zip(candidate.iter()).all(|(a, b)| (*a).is_close(*b))
                }) {
                    continue;
                }

                let min_dist = self.min_distance_to_selected(candidate);
                if min_dist > best_min_distance {
                    best_min_distance = min_dist;
                    best_candidate_idx = idx;
                }
            }

            result.push(candidates[best_candidate_idx].clone());
            self.selected_points.push(candidates[best_candidate_idx].clone());
        }

        result
    }
}

impl<R: RealScalar> Default for GreedyFurthestPoint<R> {
    fn default() -> Self {
        Self::new(
            R::from64(-std::f64::consts::PI),
            R::from64(std::f64::consts::PI),
            10000, // Default number of candidates
        )
    }
}

impl<R: RealScalar> InitialGuessGenerator<R> for GreedyFurthestPoint<R> {
    fn generate(&self, num_params: usize) -> Vec<R> {
        let mut generator = GreedyFurthestPoint::new(
            self.lower_bound,
            self.upper_bound,
            self.num_candidates,
        );
        
        // Copy existing selected points
        generator.selected_points = self.selected_points.clone();
        
        let candidates = generator.generate_candidates(num_params);
        
        if generator.selected_points.is_empty() {
            // First point - select randomly
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..candidates.len());
            return candidates[idx].clone();
        }

        // Find furthest point from all selected points
        let mut best_candidate = candidates[0].clone();
        let mut best_min_distance = R::zero();

        for candidate in &candidates {
            let min_dist = generator.min_distance_to_selected(candidate);
            if min_dist > best_min_distance {
                best_min_distance = min_dist;
                best_candidate = candidate.clone();
            }
        }

        best_candidate
    }
}
