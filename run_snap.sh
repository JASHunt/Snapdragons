python3 -c "
from snapdragons.pipeline import Pipeline
from snapdragons.config import Config

cfg = Config(
    snapshot_path='data_examples/AMNH.010000',
    param_file='data_examples/disklow.param',
    output_path='output/catalogue_ext.h5',
    output_format='hdf5',
    gaia_release='dr3',
    mag_limit=10.0,
    extinction_mode='mwdust',
    imf='kroupa',
    sampling='mass_budget',
    apply_tform_correction=True,
    fsample=1.0,
    random_seed=42,
    output_coords=['equatorial', 'galactic', 'cartesian'],
    make_plots=True,
    plot_dir='output/plots/extinction',
    logage_min=6.6,
    logage_max=10.1,
    logage_step=0.1,
    mh_min=-2.0,
    mh_max=0.5,
    mh_step=0.1,
)

pipe = Pipeline(cfg)
pipe.run()
" 2>&1
