use crate::game::Game;
use candle_core::{Device, Result, Tensor};
use tracing::debug;

pub fn get_model_input_from_game<const W: usize, const L: usize>(
    game: &Game<W, L>,
    device: &Device,
) -> Result<Tensor> {
    let state: Vec<f32> = game
        .get_state()
        .into_iter()
        .map(|a| {
            let mut state: [f32; 3] = [0.0, 0.0, 0.0];
            if *a == 0 {
                return state;
            }
            state[*a as usize - 1] = 1.0;
            return state;
        })
        .flatten()
        .collect();
    let tensor = Tensor::from_vec(state, (W, L, 3), &device)?;
    // debug!("Shape: {:?} Before {:?}", tensor.shape(),tensor.to_vec3::<f32>()?);
    let output_tensor = tensor.transpose(0, 2)?.transpose(1, 2)?;

    // debug!("Shape: {:?} After {:?}", output_tensor.shape(),output_tensor.to_vec3::<f32>()?);

    return Ok(output_tensor);
}
