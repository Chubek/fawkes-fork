
use pyo3::prelude::*;
use dssim::*;
use rgb::*;

#[pyfunction]
fn get_dissim_map_and_sim_score(
    source_image: Vec<Vec<Vec<u8>>>,
    target_image: Vec<Vec<Vec<u8>>>,
) -> PyResult<Vec<(f64, Vec<Vec<f32>>)>> {
    let width = source_image.len();
    let height = source_image[0].len();

    let vec_slice_source = source_image.into_iter()
                        .map(|x| {
                            x.into_iter()
                                .map(|l| {
                                    let s = l.as_slice();

                                    s.as_rgb()[0]
                                })
                        })
                        .flatten()
                        .collect::<Vec<RGB<u8>>>();

    let vec_slice_target = target_image.into_iter()
                    .map(|x| {
                        x.into_iter()
                            .map(|l| {
                                let s = l.as_slice();

                                s.as_rgb()[0]
                            })
                    })
                    .flatten()
                    .collect::<Vec<RGB<u8>>>();

    

    let dssim = Dssim::new();

    let image_source = dssim.create_image_rgb(vec_slice_source.as_slice(), width, height).unwrap();
    let image_target = dssim.create_image_rgb(vec_slice_target.as_slice(), width, height).unwrap();

    let (_, map) = dssim.compare(&image_source, &image_target);

    let maps_and_scores_tuple: Vec<_> = map.into_iter()
                            .map(|x|  {
                                let ssim_score = x.ssim;
                                let map = x.map;

                                let v: Vec<_> = map
                                                    .rows()
                                                    .map(|x| x.to_vec())
                                                    .collect();

                                (ssim_score, v)
                            })
                            .collect::<_>();

    Ok(maps_and_scores_tuple)
}

#[pymodule]
fn fawkes(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_dissim_map_and_sim_score, m)?)?;
    Ok(())
}