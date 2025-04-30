// #![warn(missing_docs)]

//! QuditCircuit is a library for simulating and instantiating qudit quantum
//! circuits.
// pub mod inst;
pub mod circuit;
pub mod gates;
mod inst;
pub mod math;
mod radices;
pub mod sim;
mod system;

pub use circuit::QuditCircuit;
pub use gates::Gate;
pub use math::perm::QuditPermutation;
pub use radices::*;
pub use system::QuditSystem;

#[cfg(test)]
mod tests {

    // use std::time::Instant;

    use std::time::Instant;

    // use aligned_vec::CACHELINE_ALIGN;
    use faer_core::Mat;

    use crate::circuit::CircuitLocation;
    use crate::circuit::QuditCircuit;
    use crate::loc;
    use crate::math::Function;
    // use crate::math::c32;
    use crate::math::c64;
    use crate::math::matrix::MatGrad;
    // use crate::math::unitary::DifferentiableUnitaryFn;
    use crate::radices;
    use crate::radices::QuditRadices;
    use crate::sim::BufferReuser;
    use crate::sim::BytecodeGenerator;
    use crate::sim::ExpressionTree;
    use crate::sim::QVMType;
    use crate::sim::StaticBytecodeOptimizer;
    use crate::sim::QVM;
    use crate::Gate;
    // use crate::sim::BufferOptimizer;
    use crate::sim::compile;
    use crate::sim::TreeOptimizer;

    pub fn build_qsearch_single_step_circuit(n: usize) -> QuditCircuit {
        let mut circ = QuditCircuit::new(radices![2; n], 0);
        for i in 0..n {
            circ.append_gate(Gate::U3(), loc![i], vec![]);
        }
        for i in 0..(n - 1) {
            circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
            circ.append_gate(Gate::U3(), loc![i], vec![]);
            circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
        }
        circ
    }

    pub fn build_qsearch_thin_step_circuit(n: usize) -> QuditCircuit {
        let mut circ = QuditCircuit::new(radices![2; n], 0);
        for i in 0..n {
            circ.append_gate(Gate::U3(), loc![i], vec![]);
        }
        for _ in 0..n {
            for i in 0..(n - 1) {
                circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
                circ.append_gate(Gate::U3(), loc![i], vec![]);
                circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
            }
        }
        circ
    }

    pub fn build_qsearch_thick_step_circuit(n: usize) -> QuditCircuit {
        let mut circ = QuditCircuit::new(radices![2; n], 0);
        for i in 0..n {
            circ.append_gate(Gate::U3(), loc![i], vec![]);
        }
        for _ in 0..n {
            for i in 0..(n - 1) {
                for _j in 0..3 {
                    circ.append_gate(Gate::CX(), loc![i, i + 1], vec![]);
                    circ.append_gate(Gate::U3(), loc![i], vec![]);
                    circ.append_gate(Gate::U3(), loc![i + 1], vec![]);
                }
            }
        }
        circ
    }

    use crate::sim::tree::ContractNode;
    use crate::sim::tree::KronNode;
    use crate::sim::tree::MulNode;

    #[test]
    fn custom_opt_test() {
        let cx = ExpressionTree::Leaf(Gate::CX());
        let u3 = ExpressionTree::Leaf(Gate::U3());
        let kron = ExpressionTree::Kron(KronNode::new(u3.clone(), u3.clone()));
        let block1 =
            ExpressionTree::Mul(MulNode::new(cx.clone(), kron.clone()));
        let block2 =
            ExpressionTree::Mul(MulNode::new(block1.clone(), block1.clone()));
        let block =
            ExpressionTree::Mul(MulNode::new(block2.clone(), block1.clone()));

        let left_block =
            ExpressionTree::Mul(MulNode::new(kron.clone(), block.clone()));
        let left_semi_block = ExpressionTree::Contract(ContractNode::new(
            u3.clone(),
            block.clone(),
            vec![2],
            vec![1, 2],
        ));
        let down_stair2 = ExpressionTree::Contract(ContractNode::new(
            left_block.clone(),
            left_semi_block.clone(),
            vec![0, 1],
            vec![1, 2],
        ));
        let top_tri_left = ExpressionTree::Contract(ContractNode::new(
            down_stair2.clone(),
            block.clone(),
            vec![0, 1, 2],
            vec![0, 1],
        ));

        let down_stair2 = ExpressionTree::Contract(ContractNode::new(
            block.clone(),
            block.clone(),
            vec![0, 1],
            vec![1, 2],
        ));
        let top_tri = ExpressionTree::Contract(ContractNode::new(
            down_stair2.clone(),
            block.clone(),
            vec![0, 1, 2],
            vec![0, 1],
        ));

        let left_semi_block = ExpressionTree::Contract(ContractNode::new(
            u3.clone(),
            block.clone(),
            vec![4],
            vec![3, 4],
        ));
        let up_stair2_left = ExpressionTree::Contract(ContractNode::new(
            left_semi_block.clone(),
            block.clone(),
            vec![3, 4],
            vec![2, 3],
        ));
        let bot_tri_left = ExpressionTree::Contract(ContractNode::new(
            up_stair2_left.clone(),
            block.clone(),
            vec![2, 3, 4],
            vec![3, 4],
        ));

        let up_stair2 = ExpressionTree::Contract(ContractNode::new(
            block.clone(),
            block.clone(),
            vec![3, 4],
            vec![2, 3],
        ));
        let bot_tri = ExpressionTree::Contract(ContractNode::new(
            up_stair2.clone(),
            block.clone(),
            vec![2, 3, 4],
            vec![3, 4],
        ));

        let left_semi_block = ExpressionTree::Contract(ContractNode::new(
            u3.clone(),
            block.clone(),
            vec![3],
            vec![2, 3],
        ));
        let up_stair2 = ExpressionTree::Contract(ContractNode::new(
            left_semi_block.clone(),
            block.clone(),
            vec![2, 3],
            vec![1, 2],
        ));
        let up_stair3 = ExpressionTree::Contract(ContractNode::new(
            up_stair2.clone(),
            block.clone(),
            vec![1, 2, 3],
            vec![0, 1],
        ));

        let down_stair2 = ExpressionTree::Contract(ContractNode::new(
            block.clone(),
            block.clone(),
            vec![1, 2],
            vec![2, 3],
        ));
        let down_stair3 = ExpressionTree::Contract(ContractNode::new(
            down_stair2.clone(),
            block.clone(),
            vec![1, 2, 3],
            vec![3, 4],
        ));

        let left_tree = ExpressionTree::Contract(ContractNode::new(
            top_tri_left.clone(),
            up_stair3.clone(),
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
        ));
        let bot_tree = ExpressionTree::Contract(ContractNode::new(
            bot_tri_left.clone(),
            down_stair2.clone(),
            vec![2, 3, 4],
            vec![1, 2, 3],
        ));
        let right_tree = ExpressionTree::Contract(ContractNode::new(
            bot_tri.clone(),
            down_stair3.clone(),
            vec![2, 3, 4],
            vec![1, 2, 3, 4],
        ));
        let left = ExpressionTree::Contract(ContractNode::new(
            left_tree.clone(),
            bot_tree.clone(),
            vec![0, 1, 2, 3],
            vec![1, 2, 3, 4],
        ));
        let right = ExpressionTree::Contract(ContractNode::new(
            top_tri.clone(),
            right_tree.clone(),
            vec![0, 1, 2],
            vec![1, 2, 3, 4],
        ));
        let tree =
            ExpressionTree::Mul(MulNode::new(left.clone(), right.clone()));

        let tree = TreeOptimizer::new().optimize(tree);
        println!("{:?}", tree);
        let code = compile(&tree);
        let mut qvm: QVM<c64> = QVM::new(code, QVMType::UnitaryAndGradient);
        let params = vec![1.7; tree.get_num_params()];
        let mut unitary: Mat<c64> = Mat::identity(32, 32);
        let mut gradient: MatGrad<c64> =
            MatGrad::zeros(tree.get_num_params(), 32, 32);
        let now = Instant::now();
        for _ in 0..1000 {
            // let _ = qvm.get_unitary_and_gradient(&params);
            qvm.write_unitary_and_gradient(
                &params,
                unitary.as_mut(),
                gradient.as_mut(),
            );
        }
        let elapsed = now.elapsed();
        println!("QVM Time: {:.2?}", elapsed);
        println!("QVM Avg Time: {:.2?}", elapsed / 1000);
    }

    #[test]
    fn test_one() {
        // let circuit = build_qsearch_thick_step_circuit(5);
        let circuit = build_qsearch_thin_step_circuit(5);
        // let circuit = build_qsearch_single_step_circuit(3);
        let tree = circuit.expression_tree();
        println!("{:?}", tree);
        let code = compile(&tree);
        println!("{:?}\n\n", code);
        let code = BufferReuser::new().reuse_buffers(code);
        let mut qvm: QVM<c64> = QVM::new(code, QVMType::UnitaryAndGradient);
        let params = vec![1.7; circuit.get_num_params()];
        let mut unitary: Mat<c64> = Mat::identity(32, 32);
        let mut gradient: MatGrad<c64> =
            MatGrad::zeros(circuit.get_num_params(), 32, 32);
        let now = Instant::now();
        for _ in 0..1000 {
            // let _ = qvm.get_unitary_and_gradient(&params);
            qvm.write_unitary_and_gradient(
                &params,
                unitary.as_mut(),
                gradient.as_mut(),
            );
        }
        let elapsed = now.elapsed();
        println!("QVM Time: {:.2?}", elapsed);
        println!("QVM Avg Time: {:.2?}", elapsed / 1000);
    }

    #[test]
    fn test_two() {
        let mut circuit: QuditCircuit<c64> =
            QuditCircuit::new(radices![2, 2, 2, 2], 0);
        let cz = Gate::CZ();
        let rz = Gate::P(2);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(cz.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cz.clone(), loc![0, 2], vec![]);
        circuit.append_gate(cz.clone(), loc![0, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![1], vec![0.0]);
        circuit.append_gate(cz.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cz.clone(), loc![1, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![2], vec![0.0]);
        circuit.append_gate(cz.clone(), loc![2, 3], vec![]);
        let params = vec![1.7, 2.7, 3.7];
        let circ_op = circuit.expression_tree();
        println!("{:?}", circ_op);
        let code = BytecodeGenerator::new().generate(&circ_op);
        let code = StaticBytecodeOptimizer::new(code).optimize();
        println!("{:?}", code);
        // code.print_buffers();
        let mut qvm: QVM<c64> = QVM::new(code, QVMType::UnitaryAndGradient);
        let mut unitary: Mat<c64> = Mat::identity(16, 16);
        let mut gradient: MatGrad<c64> = MatGrad::zeros(3, 16, 16);
        qvm.write_unitary_and_gradient(
            &params,
            unitary.as_mut(),
            gradient.as_mut(),
        );
        println!("{:?}", unitary);
    }

    #[test]
    fn test_three() {
        let mut circuit: QuditCircuit<c64> =
            QuditCircuit::new(radices![2, 2, 2, 2], 0);
        let cx = Gate::CZ();
        let rz = Gate::P(2);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(cx.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cx.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cx.clone(), loc![2, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(cx.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cx.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cx.clone(), loc![2, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(cx.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cx.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cx.clone(), loc![2, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(cx.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cx.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cx.clone(), loc![2, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(cx.clone(), loc![0, 1], vec![]);
        circuit.append_gate(cx.clone(), loc![1, 2], vec![]);
        circuit.append_gate(cx.clone(), loc![2, 3], vec![]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![0], vec![0.0]);
        circuit.append_gate(rz.clone(), loc![1], vec![1.0]);
        circuit.append_gate(rz.clone(), loc![2], vec![2.0]);
        circuit.append_gate(rz.clone(), loc![3], vec![2.0]);
        // assert_eq!(Array2::<c64>::eye(16), circuit.get_unitary(&[]));

        let circ_op = circuit.expression_tree();
        println!("{:?}", circ_op);
        let code = BytecodeGenerator::new().generate(&circ_op);
        let code = StaticBytecodeOptimizer::new(code).optimize();
        println!("{:?}", code);
        // code.print_buffers();
        let mut qvm: QVM<c64> = QVM::new(code, QVMType::UnitaryAndGradient);
        let mut unitary: Mat<c64> = Mat::identity(16, 16);
        let mut gradient: MatGrad<c64> = MatGrad::zeros(72, 16, 16);

        let params = vec![
            0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0,
            2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0,
            2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0,
        ];

        let now = Instant::now();
        for _ in 0..1000 {
            qvm.write_unitary_and_gradient(
                &params,
                unitary.as_mut(),
                gradient.as_mut(),
            );
        }
        let elapsed = now.elapsed();
        println!("QVM Time: {:.2?}", elapsed);

        // let circ_op: ExpressionTree<c32> = circuit.expression_tree();
        // println!("{:?}", circ_op);
        // // println!("{:?}", circ_op);
        // for i in 0..1000 {
        //     // let data = circ_op.get_unitary_and_gradient(&[
        //     let data = circ_op.get_unitary_and_gradient(&[
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         // 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         // 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0,
        //         // 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0,
        //     ]);
        // }

        // let now = Instant::now();
        // for i in 0..100 {
        //     let data = circuit.get_unitary_and_gradient(&[
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        //         0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0,
        //         2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0,
        //     ]);
        // }
        // let elapsed = now.elapsed();
        // println!("Circuit Time: {:.2?}", elapsed);
        // assert_eq!(Array2::<c64>::eye(16), circ_op.get_unitary(&[]));
    }

    //     #[test]
    //     fn test_four() {
    //         let mut circuit = QuditCircuit::new(radices![2, 2, 2, 2]);
    //         let cx = Gate::CX;
    //         let rz = Gate::RZ;
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![0]),
    // vec![0.0]);         circuit.append_gate(&rz,
    // CircuitLocation::new(vec![1]), vec![1.0]);         circuit.
    // append_gate(&rz, CircuitLocation::new(vec![2]), vec![2.0]);
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![3]),
    // vec![3.0]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![0, 1]), vec![]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![0, 1]), vec![]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![2, 3]),
    // vec![]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![2, 3]), vec![]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![1, 2]), vec![]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![1, 2]),
    // vec![]);         circuit.append_gate(&rz,
    // CircuitLocation::new(vec![0]), vec![0.0]);         circuit.
    // append_gate(&rz, CircuitLocation::new(vec![1]), vec![1.0]);
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![2]),
    // vec![2.0]);         circuit.append_gate(&rz,
    // CircuitLocation::new(vec![3]), vec![3.0]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![0, 1]), vec![]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![0, 1]),
    // vec![]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![2, 3]), vec![]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![2, 3]), vec![]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![1, 2]),
    // vec![]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![1, 2]), vec![]);         circuit.
    // append_gate(&rz, CircuitLocation::new(vec![0]), vec![0.0]);
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![1]),
    // vec![1.0]);         circuit.append_gate(&rz,
    // CircuitLocation::new(vec![2]), vec![2.0]);         circuit.
    // append_gate(&rz, CircuitLocation::new(vec![3]), vec![3.0]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![0, 1]),
    // vec![]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![0, 1]), vec![]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![2, 3]), vec![]);
    //         circuit.append_gate(&cx, CircuitLocation::new(vec![2, 3]),
    // vec![]);         circuit.append_gate(&cx,
    // CircuitLocation::new(vec![1, 2]), vec![]);         circuit.
    // append_gate(&cx, CircuitLocation::new(vec![1, 2]), vec![]);
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![0]),
    // vec![0.0]);         circuit.append_gate(&rz,
    // CircuitLocation::new(vec![1]), vec![1.0]);         circuit.
    // append_gate(&rz, CircuitLocation::new(vec![2]), vec![2.0]);
    //         circuit.append_gate(&rz, CircuitLocation::new(vec![3]),
    // vec![3.0]);         // assert_eq!(Array2::<c64>::eye(16),
    // circuit.get_unitary(&[]));         let circ_op =
    // circuit.to_operator();

    //         let now = Instant::now();
    //         for i in 0..1000 {
    //             let (utry, grad) = circ_op.get_unitary_and_gradient(&[0.0,
    // 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0,
    // 3.0]);         }
    //         let elapsed = now.elapsed();
    //         println!("Time: {:.2?}", elapsed);
    //         // assert_eq!(Array2::<c64>::eye(16), circ_op.get_unitary(&[]));
    //     }

    //     // #[test]
    //     // fn test_three() {
    //     //     let mut circuit = Circuit::new(2, vec![2, 2]);
    //     //     let gate = Gate::CX;
    //     //     circuit.append_gate(&gate, CircuitLocation::new(vec![0, 1]),
    // vec![]);     //     circuit.append_gate(&gate,
    // CircuitLocation::new(vec![0, 1]), vec![]);     //     let one =
    // c64::new(1.0, 0.0);     //     let zero = c64::new(0.0, 0.0);
    //     //     assert_eq!(
    //     //         Array2::from_shape_vec(
    //     //             (4, 4),
    //     //             vec![
    //     //                 one, zero, zero, zero, zero, one, zero, zero,
    // zero, zero, one, zero, zero,     //                 zero, zero, one,
    //     //             ],
    //     //         )
    //     //         .unwrap(),
    //     //         circuit.get_unitary(&[])
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_four() {
    //     //     let mut circuit = Circuit::new(2, vec![2, 2]);
    //     //     let gate = Gate::CX;
    //     //     circuit.append_gate(&gate, CircuitLocation::new(vec![0, 1]),
    // vec![]);     //     circuit.append_gate(&gate,
    // CircuitLocation::new(vec![0, 1]), vec![]);     //     let op =
    // &circuit[0];     //     // println!("{:?}", op);
    //     //     assert_eq!(op.get_next_on(0).unwrap(), 1);
    //     //     let next_ops = circuit.next(op);
    //     //     assert_eq!(next_ops[&0], next_ops[&1]);
    //     //     // println!("{:?}", next_ops[&0]);
    //     // }
}
