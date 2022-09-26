from typing import List
from fawkes.fork.face_op import FaceBase
from fawkes.fork.optimizer import Optimizer
from fawkes.fork.target import find_furthest_cluster




def fawkes_main(
    source_paths: List[str], 
    lr=1e-7,
    max_iter=500
) -> List[str]:
    img_sources = [FaceBase.load_and_new(p) for p in source_paths]
    img_targets = find_furthest_cluster(img_sources)

    optimizer = Optimizer.new(
        img_sources, 
        img_targets,
        lr
    )

    optimizer.fit(max_iter=max_iter)

    fin_src_imgs = optimizer.source_images

    [f.save_img("/home/chubak/Pictures/obama_fawkes.png") for f in fin_src_imgs]

    #return [f.img_to_b64_urlencoded() for f in fin_src_imgs]


fawkes_main(["/home/chubak/Pictures/obama.png"])
