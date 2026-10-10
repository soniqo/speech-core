// Reference-only binary, compiled inside a pinned upstream DeepFilterNet checkout.
// Uses the actual DfTract::process(), including conditional decoder state updates.
use std::{env, fs, path::PathBuf};
use anyhow::{bail, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 9 {
        bail!("usage: reference bundle.tar.gz input.f32 output.f32 min_snr max_erb max_df pf_beta atten_db");
    }
    let params = RuntimeParams::default()
        .with_thresholds(args[4].parse()?, args[5].parse()?, args[6].parse()?)
        .with_post_filter(args[7].parse()?)
        .with_atten_lim(args[8].parse()?);
    let mut runtime = DfTract::new(DfParams::new(PathBuf::from(&args[1]))?, &params)?;
    let bytes = fs::read(&args[2])?;
    if bytes.len() % 4 != 0 { bail!("input is not Float32 PCM"); }
    let input: Vec<f32> = bytes.chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
    if input.is_empty() { bail!("input must not be empty"); }
    let hop = runtime.hop_size;
    let bypass = params.atten_lim_db.abs() < 0.01;
    let latency = if bypass { hop } else { hop + runtime.fft_size - hop + runtime.lookahead * hop };
    // Match the C++ arbitrary-packet adapter's single-hop buffering. Every
    // following hop comes directly from Rust; no Python replica of its DSP.
    let mut output = vec![0.0f32; hop];
    let total = input.len() + latency;
    let mut offset = 0;
    let mut stages = [0usize; 4];
    while output.len() < total {
        let mut frame = vec![0.0; hop];
        let count = hop.min(input.len().saturating_sub(offset));
        frame[..count].copy_from_slice(&input[offset..offset + count]);
        offset += count;
        let mut clean = vec![0.0; hop];
        let snr = runtime.process(ArrayView2::from_shape((1, hop), &frame)?,
                                  ArrayViewMut2::from_shape((1, hop), &mut clean)?)?;
        let (gains, zeros, filter) = runtime.apply_stages(snr);
        stages[if zeros { 0 } else if !gains { 1 } else if !filter { 2 } else { 3 }] += 1;
        output.extend(clean);
    }
    output.truncate(total);
    let output: Vec<u8> = output.iter().flat_map(|sample| sample.to_le_bytes()).collect();
    fs::write(&args[3], output)?;
    println!("Rust stages [zero mask, bypass, ERB only, ERB+DF]: {stages:?}");
    Ok(())
}
