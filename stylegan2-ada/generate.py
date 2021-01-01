# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re
import imageio

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

from tqdm import tqdm

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:

        with open(os.path.join(dlatents_npz, 'final_selection'), 'r') as f:
            sel_list = f.readlines()

        dlatents_npz = [os.path.join(dlatents_npz, f'{int(i)}.npz') for i in sel_list] # [os.path.join(dlatents_npz, f) for f in os.listdir(dlatents_npz) if '.npz' in f]

        frames = 60
        scale = 4

        writer = imageio.get_writer(f'''{outdir}/{dlatents_npz[0].split('/')[-1]}.mp4''', mode='I', fps=frames, codec='libx264', bitrate='16M')

        for dlt1, dlt2 in tqdm(zip(dlatents_npz[:-1], dlatents_npz[1:])):
            print(f'Generating images from dlatents file "{dlatents_npz}"')
            dlatents1 = np.load(dlt1)['dlatents']
            dlatents2 = np.load(dlt2)['dlatents']

            step = (dlatents2 - dlatents1) / (scale*frames)

            for j in range(scale*frames):
                dlatents = dlatents1 + step * j
                # assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
                imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
                for i, img in enumerate(imgs):
                    fname = f'''{outdir}/dlatent_{dlt1.split('/')[-1]}_{j}_{i:02d}.png'''
                    # PIL.Image.fromarray(img, 'RGB').save(fname)
                    writer.append_data(img)

        writer.close()
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]

        dlatents = Gs.components.mapping.run(z, None).astype(np.float32)  # [N, L, C]
        # dlatents = dlatents[:, :1, :].astype(np.float32)
        np.savez(f'{outdir}/{seed}.npz', dlatents=dlatents)

        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
