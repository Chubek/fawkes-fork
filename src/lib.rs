
use pyo3::prelude::*;
use dssim::*;
use rgb::*;


#[pyfunction]
fn get_dissim_map_and_sim_score(
    source_image: Vec<Vec<Vec<u8>>>,
    target_image: Vec<Vec<Vec<u8>>>,
) -> PyResult<(f64, Vec<Vec<Vec<f32>>>)>{

    let width_src = source_image.len();
    let height_src = source_image[0].len();

    let width_target = target_image.len();
    let height_target = target_image[0].len();

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

    

    let mut dssim = Dssim::new();
    

    dssim.set_save_ssim_maps(20);


    let image_source = dssim.create_image_rgb(
        vec_slice_source.as_slice(), width_src, height_src).unwrap();
    let image_target = dssim.create_image_rgb(
        vec_slice_target.as_slice(), width_target, height_target).unwrap();
    
    let (score, map) = dssim.compare(&image_source, &image_target);
   
    let maps_and_scores_tuple: Vec<_> = map.into_iter()
                            .map(|x|  {
                                let mapi = x.map;
                                
                                let v: Vec<_> = mapi
                                                    .rows()
                                                    .map(|x| x.to_vec())
                                                    .collect();

                                v
                            })
                            .collect::<_>();

    Ok((score.into(), maps_and_scores_tuple))
}

#[pymodule]
fn fawkes_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_dissim_map_and_sim_score, m)?)?;
    Ok(())
}