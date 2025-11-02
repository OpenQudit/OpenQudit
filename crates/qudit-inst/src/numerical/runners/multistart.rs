use qudit_core::RealScalar;
use super::super::Runner;
use super::super::Problem;
use super::super::InitialGuessGenerator;
use super::super::MinimizationAlgorithm;
use super::super::MinimizationResult;

#[derive(Clone)]
pub struct MultiStartRunner<M, G> {
    pub minimizer: M,
    pub guess_generator: G,
    pub num_starts: usize,
}

impl<R, P, M, G> Runner<R, P> for MultiStartRunner<M, G>
where
    R: RealScalar,
    P: Problem,
    M: MinimizationAlgorithm<R, P>,
    G: InitialGuessGenerator<R>,
{
    fn run(&self, problem: P) -> MinimizationResult<R> {
        let mut best: Option<MinimizationResult<R>> = None;
        let mut func = self.minimizer.initialize(&problem);
        for _ in 0..self.num_starts {
            let x0 = self.guess_generator.generate(problem.num_params());
            // let now = std::time::Instant::now();
            // for _ in 0..100 {
            //     let res = self.minimizer.minimize(&mut func, &x0);
            // }
            // let elapsed = now.elapsed();
            // println!("Minimization took: {:?}", elapsed/100);

            let res = self.minimizer.minimize(&mut func, &x0);
            if res.fun <= R::from64(1e-4) {
                return res;
            }
            // println!("{}", res.fun);
            match &mut best {
                None => best = Some(res),
                Some(b) if res.fun < b.fun => *b = res,
                _ => {}
            }
        }
        best.expect("MultiStart: num_starts==0")
    }
}

// impl<C, P, M, F, G> Runner<C, P, F> for BatchedMultiStartRunner<M, G, C>
// where
//     C: ComplexScalar,
//     P: Problem + ProvidesBulk,
//     M: MinimizationAlgorithm<C, P, F>,
//     F: Function<C>,
//     G: InitialGuessGenerator<C>,
// {
//     fn run(&self, problem: P) -> MinimizationResult<C> {
//         let mut best: Option<MinimizationResult<C>> = None;
//         let func = self.minimizer.prepare_bulk(&problem, num_starts);
//         let x0 = self.guess_generator.generate_bulk(problem.num_params(), num_starts);
//         let res = self.minimizer.execute_bulk(&func, &x0);
//             match &mut best {
//                 None => best = Some(res),
//                 Some(b) if res.fun < b.fun => *b = res,
//                 _ => {}
//             }
//         }
//         best.expect("MultiStart: num_starts==0")
//     }
// }

// ParallelStartRunner; EnsembleRunner; IterativeRefinementRunner/ChainedRunner;
// HyperTunningRunner; BenchmarkingRunner; BatchedMultiStart
